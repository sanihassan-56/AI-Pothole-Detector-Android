
package com.sani.potholedetector

import android.content.Context
import android.widget.TextView
import android.media.Image
import android.widget.ImageView
import android.media.MediaMetadataRetriever
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.os.Build
import android.view.View
import android.widget.ScrollView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import android.util.Log
import java.util.Locale
import java.util.concurrent.Executors
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import android.media.MediaMuxer
import android.widget.Button
import android.content.ContentValues
import android.provider.MediaStore
import android.widget.Toast
import android.widget.VideoView
import android.media.MediaCodecList
import android.media.MediaFormat
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.widget.PopupMenu
import android.view.Surface
import android.opengl.EGL14
import android.opengl.EGLExt
import android.opengl.GLES20
import android.content.pm.PackageManager
import org.json.JSONArray
import org.json.JSONObject

import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

data class PotholeRecord(
    val confidence: Float,
    val timestamp: Long
)

class MainActivity : AppCompatActivity() {

    companion object {
        private const val DETECTION_FPS = 10
        private const val OUTPUT_VIDEO_FPS = 25
        private const val VIDEO_BITRATE = 4_000_000
    }
    private lateinit var imagePreview: android.widget.ImageView
    private lateinit var tvImagePlaceholder: android.widget.TextView
    private lateinit var tvStatus: android.widget.TextView
    private lateinit var tvDetectionResults: android.widget.TextView
    private lateinit var btnSelectVideo: android.widget.Button
    private lateinit var btnSaveDetection: Button
    private lateinit var btnSelectImage: android.widget.Button
    private lateinit var btnDetectImage: android.widget.Button
    private lateinit var cameraPreview: PreviewView
    private lateinit var btnRealtimeCamera: Button
    private lateinit var detectionOverlay: DetectionOverlay
    private var cameraProvider: ProcessCameraProvider? = null

    //GPS Record storage var
    private val detectionRecords = mutableListOf<PotholeRecord>()
    private var latestDetectionText = "No detection results yet"
    private var latestConfidenceText = "Confidence: --"
    private var lastPotholeRecordTime = 0L
    private val potholeRecordCooldown = 10_000L
    //Gps record Button
    private lateinit var btnViewHistory: Button

    // Annotated video output
    private var annotatedVideoFile: File? = null

    // Video playback
    private lateinit var videoPreview: VideoView

    // Video processing state
    private var isVideoProcessing = false
    private var selectedBitmap: Bitmap? = null
    private var latestAnnotatedBitmap: Bitmap? = null
    private var engine: PotholeOnnxEngine? = null
    private lateinit var btnPlayAnnotatedVideo: Button
    private lateinit var btnSaveAnnotatedVideo: Button

    // Performs model loading and inference away from the UI thread.
    private val executor = Executors.newSingleThreadExecutor()

    private val imagePicker =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->

            if (uri != null) {
                loadSelectedImage(uri)
            }
        }
    private val videoPicker =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->

            if (uri != null) {
                processVideo(uri)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        loadPotholeRecords()
        showHomeDashboard()
        initializeModel()
    }


    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {

        try {
            val buffer = image.planes[0].buffer
            buffer.rewind()

            val pixelStride = image.planes[0].pixelStride
            val rowStride = image.planes[0].rowStride

            val width = image.width
            val height = image.height

            val rowPadding = rowStride - pixelStride * width

            val bitmapWidth =
                width + rowPadding / pixelStride

            val bitmap = Bitmap.createBitmap(
                bitmapWidth,
                height,
                Bitmap.Config.ARGB_8888
            )

            bitmap.copyPixelsFromBuffer(buffer)

            val croppedBitmap = Bitmap.createBitmap(
                bitmap,
                0,
                0,
                width,
                height
            )

            if (croppedBitmap !== bitmap) {
                bitmap.recycle()
            }

            val rotation = image.imageInfo.rotationDegrees

            return if (rotation != 0) {

                val matrix = android.graphics.Matrix()
                matrix.postRotate(rotation.toFloat())

                val rotatedBitmap = Bitmap.createBitmap(
                    croppedBitmap,
                    0,
                    0,
                    croppedBitmap.width,
                    croppedBitmap.height,
                    matrix,
                    true
                )

                croppedBitmap.recycle()
                rotatedBitmap

            } else {
                croppedBitmap
            }

        } catch (e: Exception) {
            Log.e("POTHOLE_AI", "Bitmap conversion failed", e)
            return null
        }
    }

    private val cameraAnalysisExecutor =
        Executors.newSingleThreadExecutor()

    private var lastInferenceTime = 0L

    private fun startCamera() {


        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({

            try {
                cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder()
                    .build()

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(
                        ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                    )
                    .setOutputImageFormat(
                        ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
                    )
                    .build()

                imageAnalysis.setAnalyzer(
                    cameraAnalysisExecutor
                ) { imageProxy ->
                    analyzeCameraFrame(imageProxy)
                }


                preview.setSurfaceProvider(
                    cameraPreview.surfaceProvider
                )

                val cameraSelector =
                    CameraSelector.DEFAULT_BACK_CAMERA

                cameraProvider?.unbindAll()

                cameraProvider?.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )

                // Show live camera preview.
                cameraPreview.visibility = View.VISIBLE
                detectionOverlay.visibility = View.VISIBLE
                imagePreview.visibility = View.GONE
                videoPreview.visibility = View.GONE
                tvImagePlaceholder.visibility = View.GONE

                tvStatus.text = "Status: Camera started successfully."

                Log.i(
                    "POTHOLE_AI",
                    "CameraX live preview started."
                )

            } catch (e: Exception) {

                Log.e(
                    "POTHOLE_AI",
                    "Camera initialization failed",
                    e
                )

                tvStatus.text =
                    "Status: Camera error - ${e.message}"
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun initializeViews() {

        imagePreview = findViewById(R.id.imagePreview)
        tvImagePlaceholder = findViewById(R.id.tvImagePlaceholder)
        tvStatus = findViewById(R.id.tvStatus)
        tvDetectionResults = findViewById(R.id.tvDetectionResults)
        btnSelectVideo = findViewById(R.id.btnSelectVideo)
        btnSaveDetection = findViewById(R.id.btnSaveDetection)
        btnSelectImage = findViewById(R.id.btnSelectImage)
        btnDetectImage = findViewById(R.id.btnDetectImage)
        //gps initialize record button
        btnViewHistory = findViewById(R.id.btnViewHistory)
    }

    private fun setupButtons() {

        btnSelectVideo.setOnClickListener {
            videoPicker.launch("video/*")
        }

        btnSelectImage.setOnClickListener {
            imagePicker.launch("image/*")
        }

        btnDetectImage.setOnClickListener {
            runImageDetection()
        }

        btnSaveDetection.setOnClickListener {
            saveDetection()
        }
    }

    // Load the ONNX model and keep the session available for inference.

    private fun initializeModel() {

        executor.execute {
            try {
                engine = PotholeOnnxEngine(applicationContext)

                val modelInfo = engine!!.getModelInformation()

                Log.i("POTHOLE_AI", "Model loaded successfully")
                Log.i("POTHOLE_AI", modelInfo)

            } catch (e: Exception) {

                Log.e("POTHOLE_AI", "Model loading failed", e)

                runOnUiThread {
                    Toast.makeText(
                        this,
                        "Model loading failed: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }


    private fun requestCameraPermissionAndStart() {
        if (
            ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            cameraPermissionLauncher.launch(
                android.Manifest.permission.CAMERA
            )
        }
    }



    private val cameraPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->

            if (granted) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    "Camera permission is required for real-time detection.",
                    Toast.LENGTH_LONG
                ).show()
            }
        }

    // Decode the image selected from the phone.
    private fun loadSelectedImage(uri: Uri) {

        btnDetectImage.isEnabled = false
        tvStatus.text = "Status: Loading selected image..."
        tvDetectionResults.text = "Waiting for detection..."

        executor.execute {

            try {
                val bitmap = contentResolver.openInputStream(uri).use { stream ->
                    requireNotNull(stream) {
                        "Unable to open the selected image."
                    }
                    BitmapFactory.decodeStream(stream)
                } ?: throw IllegalArgumentException("Unable to decode image.")

                selectedBitmap = bitmap

                runOnUiThread {

                    imagePreview.setImageBitmap(bitmap)
                    tvImagePlaceholder.visibility = View.GONE

                    btnDetectImage.isEnabled = engine != null

                    tvStatus.text =
                        if (engine != null) {
                            "Status: Image loaded. Ready for detection."
                        } else {
                            "Status: Image loaded. Waiting for model..."
                        }
                }

            } catch (e: Exception) {

                Log.e("POTHOLE_AI", "Image loading failed", e)

                runOnUiThread {
                    tvStatus.text = "Status: Image loading failed - ${e.message}"
                    btnDetectImage.isEnabled = false
                }
            }
        }
    }

    // Run YOLO26n inference on the selected image.
    private fun runImageDetection() {

        val bitmap = selectedBitmap

        if (bitmap == null) {
            tvStatus.text = "Status: Please select an image first."
            return
        }

        btnDetectImage.isEnabled = false
        btnSelectImage.isEnabled = false

        tvStatus.text = "Status: Detecting road defects..."
        tvDetectionResults.text = "Running YOLO26n inference..."

        executor.execute {

            try {
                val detector = engine
                    ?: throw IllegalStateException("YOLO26n model is not ready.")

                val detections = detector.detect(bitmap)
                if (detections.isNotEmpty()) {
                    val bestDetection = detections.maxByOrNull {
                        it.confidence
                    }

                    if (bestDetection != null) {
                        val record = PotholeRecord(
                            confidence = bestDetection.confidence,
                            timestamp = System.currentTimeMillis()
                        )

                        detectionRecords.add(record)
                        savePotholeRecord(record)
                    }
                }

                val annotatedBitmap = drawDetections(bitmap, detections)
                if (detections.isNotEmpty()) {
                    latestAnnotatedBitmap = annotatedBitmap
                }

                val resultText = if (detections.isEmpty()) {

                    "No road defects detected."

                } else {

                    buildString {

                        append("Total detections: ${detections.size}\n\n")

                        detections.forEachIndexed { index, detection ->

                            append("${index + 1}. ${detection.className}\n")

                            append(
                                "Confidence: ${
                                    String.format(
                                        Locale.US,
                                        "%.2f",
                                        detection.confidence * 100
                                    )
                                }%\n\n"
                            )
                        }
                    }
                }

                runOnUiThread {
                    latestDetectionText = if (detections.isEmpty()) {
                        "No pothole detected"
                    } else {
                        "${detections.size} pothole(s) detected"
                    }

                    latestConfidenceText = if (detections.isEmpty()) {
                        "Confidence: --"
                    } else {
                        val highestConfidence = detections.maxOf { it.confidence }

                        "Confidence: ${String.format(
                            Locale.US,
                            "%.2f",
                            highestConfidence * 100
                        )}%"
                    }

                    imagePreview.setImageBitmap(annotatedBitmap)

                    tvDetectionResults.text = resultText

                    tvStatus.text =
                        "Status: Detection completed successfully."

                    btnDetectImage.isEnabled = true
                    btnSelectImage.isEnabled = true

                    btnSaveDetection.isEnabled = detections.isNotEmpty()
                }

                Log.i(
                    "POTHOLE_AI",
                    "Image inference completed. Detections: ${detections.size}"
                )

            } catch (e: Exception) {

                Log.e("POTHOLE_AI", "Image detection failed", e)

                runOnUiThread {

                    tvStatus.text =
                        "Status: Detection failed - ${e.message}"

                    btnDetectImage.isEnabled = true
                    btnSelectImage.isEnabled = true
                }
            }
        }
    }


    private fun inspectVideoFormat(uri: Uri) {

        val extractor = MediaExtractor()

        try {
            extractor.setDataSource(this, uri, null)

            for (i in 0 until extractor.trackCount) {

                val format = extractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: continue

                if (mime.startsWith("video/")) {

                    val width = format.getInteger(MediaFormat.KEY_WIDTH)
                    val height = format.getInteger(MediaFormat.KEY_HEIGHT)

                    val frameRate = if (
                        format.containsKey(MediaFormat.KEY_FRAME_RATE)
                    ) {
                        format.getInteger(MediaFormat.KEY_FRAME_RATE)
                    } else {
                        0
                    }

                    Log.i(
                        "POTHOLE_AI",
                        "VIDEO FORMAT: ${width}x${height}, FPS: $frameRate"
                    )
                }

                if (mime.startsWith("audio/")) {
                    Log.i("POTHOLE_AI", "Original audio track found: $mime")
                }
            }

        } catch (e: Exception) {
            Log.e("POTHOLE_AI", "Video format inspection failed", e)
        } finally {
            extractor.release()
        }
    }




    private fun createVideoDecoder(
        originalFormat: MediaFormat
    ): MediaCodec {
        val mimeType = originalFormat.getString(
            MediaFormat.KEY_MIME
        ) ?: throw IllegalStateException("Video MIME type missing.")

        val decoder = MediaCodec.createDecoderByType(mimeType)

        try {
            Log.i(
                "POTHOLE_AI",
                "SELECTED DECODER: ${decoder.name}"
            )

            Log.i(
                "POTHOLE_AI",
                "ORIGINAL VIDEO FORMAT: $originalFormat"
            )

            decoder.configure(
                originalFormat,
                null,
                null,
                0
            )

            decoder.start()

            Log.i(
                "POTHOLE_AI",
                "DECODER STARTED SUCCESSFULLY"
            )

            return decoder

        } catch (e: Exception) {
            decoder.release()
            throw e
        }
    }





    private fun createVideoEncoder(
        width: Int,
        height: Int,
        frameRate: Int
    ): MediaCodec {

        val mimeType = MediaFormat.MIMETYPE_VIDEO_AVC

        // Prefer the TECNO/MediaTek hardware encoder.
        val codecList = MediaCodecList(
            MediaCodecList.REGULAR_CODECS
        )

        val encoderInfo = codecList.codecInfos
            .filter { info ->
                info.isEncoder &&
                        info.supportedTypes.any {
                            it.equals(mimeType, ignoreCase = true)
                        }
            }
            .firstOrNull {
                it.name == "OMX.MTK.VIDEO.ENCODER.AVC"
            }
            ?: throw IllegalStateException(
                "MediaTek H.264 hardware encoder not found."
            )

        val capabilities =
            encoderInfo.getCapabilitiesForType(mimeType)

        // Use planar YUV420, which your phone supports.
        val colorFormat =
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Planar

        if (!capabilities.colorFormats.contains(colorFormat)) {
            throw IllegalStateException(
                "H.264 encoder does not support YUV420 Planar."
            )
        }

        // H.264 encoding requires even dimensions.
        if (width % 2 != 0 || height % 2 != 0) {
            throw IllegalArgumentException(
                "Video width and height must be even numbers."
            )
        }

        val format = MediaFormat.createVideoFormat(
            mimeType,
            width,
            height
        )

        format.setInteger(
            MediaFormat.KEY_COLOR_FORMAT,
            colorFormat
        )

        format.setInteger(
            MediaFormat.KEY_BIT_RATE,
            VIDEO_BITRATE
        )

        format.setInteger(
            MediaFormat.KEY_FRAME_RATE,
            frameRate
        )

        format.setInteger(
            MediaFormat.KEY_I_FRAME_INTERVAL,
            2
        )

        val encoder = MediaCodec.createByCodecName(
            encoderInfo.name
        )

        try {
            encoder.configure(
                format,
                null,
                null,
                MediaCodec.CONFIGURE_FLAG_ENCODE
            )

            Log.i(
                "POTHOLE_AI",
                "Configured H.264 encoder: ${encoderInfo.name}"
            )

            Log.i(
                "POTHOLE_AI",
                "Encoder color format: YUV420 Planar"
            )

            return encoder

        } catch (e: Exception) {
            encoder.release()
            throw e
        }
    }



    // Draw bounding boxes and class labels on a copy of the original image.





    private fun checkVideoEncoder() {

        val codecList = MediaCodecList(MediaCodecList.REGULAR_CODECS)

        val encoders = codecList.codecInfos.filter { codecInfo ->
            codecInfo.isEncoder &&
                    codecInfo.supportedTypes.any {
                        it.equals(MediaFormat.MIMETYPE_VIDEO_AVC, ignoreCase = true)
                    }
        }

        if (encoders.isEmpty()) {
            Log.e("POTHOLE_AI", "No H.264 video encoder found.")
            return
        }

        encoders.forEach { encoder ->
            Log.i(
                "POTHOLE_AI",
                "H.264 encoder: ${encoder.name}"
            )

            val capabilities = encoder.getCapabilitiesForType(
                MediaFormat.MIMETYPE_VIDEO_AVC
            )

            capabilities.colorFormats.forEach { format ->
                Log.i(
                    "POTHOLE_AI",
                    "Encoder ${encoder.name} supported color format: " +
                            String.format("0x%08X", format)
                )
            }
        }
    }


    private fun processVideo(uri: Uri) {

        btnSelectVideo.isEnabled = false
        btnSelectImage.isEnabled = false
        btnDetectImage.isEnabled = false

        tvStatus.text = "Status: Processing video..."
        tvDetectionResults.text = "Analysing video frames..."

        executor.execute {

            var videoEncoder: VideoEncoder? = null
            var decodedFrames = 0
            var framesAnalysed = 0
            var totalDetections = 0

            try {
                val detector = engine
                    ?: throw IllegalStateException("YOLO26n model is not ready.")

                inspectVideoFormat(uri)

                val summary = mutableMapOf<String, Pair<Int, Float>>()

                var lastDetections: List<Detection> = emptyList()

                val inferenceIntervalUs = 1_000_000L / DETECTION_FPS
                var nextInferenceTimeUs = 0L

                val outputFile = File(
                    cacheDir,
                    "annotated_${System.currentTimeMillis()}.mp4"
                )

                val totalFrames = decodeVideoFrames(uri) { frame, timestampUs ->

                    decodedFrames++

                    // Create the encoder using the first decoded frame dimensions.
                    if (videoEncoder == null) {

                        require(frame.width % 2 == 0 && frame.height % 2 == 0) {
                            "Video dimensions must be even for AVC encoding."
                        }

                        videoEncoder = VideoEncoder(
                            outputFile,
                            frame.width,
                            frame.height,
                            25
                        )

                        videoEncoder!!.start()
                    }

                    // Run YOLO26n at the configured inference frequency.
                    if (timestampUs >= nextInferenceTimeUs) {

                        val detections = detector.detect(frame)

                        lastDetections = detections

                        framesAnalysed++
                        totalDetections += detections.size

                        detections.forEach { detection ->

                            val previous = summary[detection.className]

                            val count = (previous?.first ?: 0) + 1

                            val highestConfidence = maxOf(
                                previous?.second ?: 0f,
                                detection.confidence
                            )

                            summary[detection.className] =
                                Pair(count, highestConfidence)
                        }

                        do {
                            nextInferenceTimeUs += inferenceIntervalUs
                        } while (nextInferenceTimeUs <= timestampUs)
                    }

                    // Annotate every frame using the latest YOLO detections.
                    val annotatedFrame = drawDetections(
                        frame,
                        lastDetections
                    )

                    try {
                        videoEncoder!!.writeFrame(
                            annotatedFrame,
                            timestampUs
                        )
                    } finally {
                        annotatedFrame.recycle()
                    }

                    // Update progress.
                    if (decodedFrames % 25 == 0) {

                        runOnUiThread {
                            tvStatus.text =
                                "Status: Processing video... " +
                                        "$decodedFrames frames decoded, " +
                                        "$framesAnalysed analysed."
                        }
                    }
                }

                val completedEncoder = videoEncoder
                    ?: throw IllegalStateException("No video frames were decoded.")

                // Finalize the MP4 file.
                completedEncoder.finish()
                videoEncoder = null

                if (!outputFile.exists() || outputFile.length() == 0L) {
                    throw IllegalStateException("Annotated video file was not created.")
                }

                annotatedVideoFile = outputFile

                val resultText = buildString {

                    append("VIDEO ANALYSIS COMPLETED\n\n")
                    append("Frames decoded: $totalFrames\n")
                    append("Frames analysed: $framesAnalysed\n")
                    append("Total detections: $totalDetections\n\n")

                    if (summary.isEmpty()) {

                        append("No potholes detected.")

                    } else {

                        append("POTHOLE DETECTION SUMMARY\n\n")

                        summary.forEach { (className, values) ->

                            append("Class: $className\n")
                            append("Frame detections: ${values.first}\n")

                            append(
                                "Highest confidence: ${
                                    String.format(
                                        Locale.US,
                                        "%.2f",
                                        values.second * 100
                                    )
                                }%\n\n"
                            )
                        }
                    }
                }
                // Save one history record for this video.
                if (summary.isNotEmpty()) {

                    val highestConfidence = summary.values.maxOf {
                        it.second
                    }

                    val record = PotholeRecord(
                        confidence = highestConfidence,
                        timestamp = System.currentTimeMillis()
                    )

                    detectionRecords.add(record)
                    savePotholeRecord(record)

                    latestDetectionText = "Pothole detected in video"
                    latestConfidenceText = "Confidence: ${
                        String.format(
                            Locale.US,
                            "%.2f",
                            highestConfidence * 100
                        )
                    }%"
                }


                runOnUiThread {

                    imagePreview.visibility = View.GONE
                    tvImagePlaceholder.visibility = View.GONE

                    videoPreview.visibility = View.VISIBLE
                    videoPreview.setVideoURI(Uri.fromFile(outputFile))

                    btnPlayAnnotatedVideo.visibility = View.VISIBLE
                    btnPlayAnnotatedVideo.isEnabled = true

                    btnPlayAnnotatedVideo.setOnClickListener {
                        videoPreview.start()
                    }

                    btnSaveAnnotatedVideo.visibility = View.VISIBLE
                    btnSaveAnnotatedVideo.isEnabled = true

                    btnSaveAnnotatedVideo.setOnClickListener {

                        val file = annotatedVideoFile

                        if (file != null) {
                            saveAnnotatedVideoToGallery(file)
                        } else {
                            Toast.makeText(
                                this,
                                "No annotated video available.",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }

                    tvDetectionResults.text = resultText

                    tvStatus.text =
                        "Status: Annotated video created successfully."

                    btnSelectVideo.isEnabled = true
                    btnSelectImage.isEnabled = true
                    btnDetectImage.isEnabled = true
                }

                Log.i(
                    "POTHOLE_AI",
                    "ANNOTATED VIDEO CREATED: $totalFrames decoded, " +
                            "$framesAnalysed analysed. File: ${outputFile.absolutePath}"
                )

            } catch (e: Exception) {

                Log.e(
                    "POTHOLE_AI",
                    "Video processing failed",
                    e
                )

                try {
                    videoEncoder?.release()
                } catch (releaseError: Exception) {
                    Log.e("POTHOLE_AI", "Encoder cleanup failed", releaseError)
                }

                videoEncoder = null

                runOnUiThread {

                    tvStatus.text =
                        "Status: Video processing failed - ${e.message}"

                    btnSelectVideo.isEnabled = true
                    btnSelectImage.isEnabled = true
                    btnDetectImage.isEnabled = true
                }
            }
        }
    }


    private fun saveAnnotatedVideoToGallery(sourceFile: File) {

        try {
            if (!sourceFile.exists()) {
                throw IllegalStateException("Annotated video file not found.")
            }

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {

                val values = ContentValues().apply {
                    put(
                        MediaStore.MediaColumns.DISPLAY_NAME,
                        "Pothole_Detection_${System.currentTimeMillis()}.mp4"
                    )

                    put(
                        MediaStore.MediaColumns.MIME_TYPE,
                        "video/mp4"
                    )

                    put(
                        MediaStore.MediaColumns.RELATIVE_PATH,
                        Environment.DIRECTORY_MOVIES + "/PotholeDetection"
                    )

                    put(MediaStore.MediaColumns.IS_PENDING, 1)
                }

                val videoUri = contentResolver.insert(
                    MediaStore.Video.Media.getContentUri(
                        MediaStore.VOLUME_EXTERNAL_PRIMARY
                    ),
                    values
                ) ?: throw IllegalStateException("Could not create gallery entry.")

                try {
                    val outputStream = contentResolver.openOutputStream(videoUri)
                        ?: throw IllegalStateException("Could not open gallery file.")

                    outputStream.use { output ->
                        sourceFile.inputStream().use { input ->
                            input.copyTo(output)
                        }
                    }

                    val completedValues = ContentValues().apply {
                        put(MediaStore.MediaColumns.IS_PENDING, 0)
                    }

                    contentResolver.update(
                        videoUri,
                        completedValues,
                        null,
                        null
                    )

                    Toast.makeText(
                        this,
                        "Annotated video saved to Movies/PotholeDetection",
                        Toast.LENGTH_LONG
                    ).show()

                } catch (e: Exception) {
                    contentResolver.delete(videoUri, null, null)
                    throw e
                }

            } else {

                val directory = getExternalFilesDir(Environment.DIRECTORY_MOVIES)
                    ?: throw IllegalStateException("Storage directory unavailable.")

                val destination = File(directory, sourceFile.name)

                sourceFile.copyTo(destination, overwrite = true)

                Toast.makeText(
                    this,
                    "Video saved to: ${destination.absolutePath}",
                    Toast.LENGTH_LONG
                ).show()
            }

        } catch (e: Exception) {

            Log.e("POTHOLE_AI", "Video saving failed", e)

            Toast.makeText(
                this,
                "Could not save video: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
        }
    }



    private fun saveDetection() {

        val bitmap = latestAnnotatedBitmap

        if (bitmap == null) {
            Toast.makeText(
                this,
                "No successful detection to save.",
                Toast.LENGTH_SHORT
            ).show()
            return
        }

        try {
            val filename = "PotholeDetection_${System.currentTimeMillis()}.jpg"

            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(
                    MediaStore.Images.Media.RELATIVE_PATH,
                    "${Environment.DIRECTORY_PICTURES}/PotholeDetector"
                )
            }

            val imageUri = contentResolver.insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                values
            ) ?: throw Exception("Unable to create image file.")

            val outputStream = contentResolver.openOutputStream(imageUri)
                ?: throw Exception("Unable to open image file.")

            outputStream.use {
                bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it)
            }

            Toast.makeText(
                this,
                "Detection image saved successfully!",
                Toast.LENGTH_LONG
            ).show()

            Log.i("POTHOLE_AI", "Saved detection image: $imageUri")

        } catch (e: Exception) {

            Log.e("POTHOLE_AI", "Image saving failed", e)

            Toast.makeText(
                this,
                "Image saving failed: ${e.message}",
                Toast.LENGTH_LONG
            ).show()
        }
    }


    private fun drawDetections(
        bitmap: Bitmap,
        detections: List<Detection>
    ): Bitmap {

        val outputBitmap =
            bitmap.copy(Bitmap.Config.ARGB_8888, true)

        val canvas = Canvas(outputBitmap)

        val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            color = Color.RED
            strokeWidth = (bitmap.width * 0.006f).coerceAtLeast(3f)
        }

        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.WHITE
            textSize = (bitmap.width * 0.035f).coerceAtLeast(18f)
            isFakeBoldText = true
        }

        val backgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.RED
        }

        detections.filter { it.classId == 2 }.forEach { detection ->

            val left = detection.left.coerceIn(0f, bitmap.width.toFloat())
            val top = detection.top.coerceIn(0f, bitmap.height.toFloat())
            val right = detection.right.coerceIn(0f, bitmap.width.toFloat())
            val bottom = detection.bottom.coerceIn(0f, bitmap.height.toFloat())

            canvas.drawRect(left, top, right, bottom, boxPaint)

            val confidence = String.format(
                Locale.US,
                "%.1f%%",
                detection.confidence * 100
            )

            val label = "Pothole $confidence"

            val labelWidth = textPaint.measureText(label) + 16f
            val labelHeight = textPaint.textSize + 12f

            val labelTop = (top - labelHeight).coerceAtLeast(0f)

            canvas.drawRect(
                left,
                labelTop,
                (left + labelWidth).coerceAtMost(bitmap.width.toFloat()),
                labelTop + labelHeight,
                backgroundPaint
            )

            canvas.drawText(
                label,
                left + 8f,
                labelTop + textPaint.textSize,
                textPaint
            )
        }

        return outputBitmap
    }


    override fun onDestroy() {
        super.onDestroy()

        // Close the ONNX session after queued work has finished.
        executor.execute {
            engine?.close()
            engine = null
        }

        executor.shutdown()
    }

    private fun bitmapToYuv420Planar(bitmap: Bitmap): ByteArray {

        val width = bitmap.width
        val height = bitmap.height

        require(width % 2 == 0 && height % 2 == 0) {
            "Bitmap dimensions must be even."
        }

        val argb = IntArray(width * height)

        bitmap.getPixels(
            argb,
            0,
            width,
            0,
            0,
            width,
            height
        )

        val frameSize = width * height
        val chromaSize = frameSize / 4

        val output = ByteArray(
            frameSize + 2 * chromaSize
        )

        var yIndex = 0
        var uIndex = frameSize
        var vIndex = frameSize + chromaSize

        for (row in 0 until height) {

            for (col in 0 until width) {

                val pixel = argb[row * width + col]

                val red = (pixel shr 16) and 0xFF
                val green = (pixel shr 8) and 0xFF
                val blue = pixel and 0xFF

                val y = (
                        (66 * red + 129 * green + 25 * blue + 128) shr 8
                        ) + 16

                output[yIndex++] = y.coerceIn(0, 255).toByte()

                if (row % 2 == 0 && col % 2 == 0) {

                    val u = (
                            (-38 * red - 74 * green + 112 * blue + 128) shr 8
                            ) + 128

                    val v = (
                            (112 * red - 94 * green - 18 * blue + 128) shr 8
                            ) + 128

                    output[uIndex++] = u.coerceIn(0, 255).toByte()
                    output[vIndex++] = v.coerceIn(0, 255).toByte()
                }
            }
        }

        return output
    }


    private fun queueEncoderFrame(
        encoder: MediaCodec,
        bitmap: Bitmap,
        presentationTimeUs: Long
    ): Boolean {
        val yuvData = bitmapToYuv420Planar(bitmap)

        val inputIndex = encoder.dequeueInputBuffer(10_000)

        if (inputIndex < 0) {
            Log.w("POTHOLE_AI", "Encoder input buffer unavailable.")
            return false
        }

        val inputBuffer = encoder.getInputBuffer(inputIndex)

        if (inputBuffer == null) {
            Log.e("POTHOLE_AI", "Encoder input buffer is null.")
            return false
        }

        if (yuvData.size > inputBuffer.capacity()) {
            Log.e(
                "POTHOLE_AI",
                "Frame too large: ${yuvData.size}, buffer: ${inputBuffer.capacity()}"
            )
            return false
        }

        inputBuffer.clear()
        inputBuffer.put(yuvData)

        encoder.queueInputBuffer(
            inputIndex,
            0,
            yuvData.size,
            presentationTimeUs,
            0
        )

        return true
    }


    private fun drainEncoder(
        encoder: MediaCodec,
        muxer: MediaMuxer,
        bufferInfo: MediaCodec.BufferInfo,
        muxerState: IntArray,
        endOfStream: Boolean = false
    ) {
        val timeoutUs = if (endOfStream) 10_000L else 0L
        var attempts = 0

        while (true) {
            val outputIndex = encoder.dequeueOutputBuffer(
                bufferInfo,
                timeoutUs
            )

            when {
                outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream) return

                    attempts++

                    if (attempts >= 500) {
                        throw IllegalStateException(
                            "Timed out waiting for encoder output."
                        )
                    }
                }

                outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    if (muxerState[0] != -1) {
                        throw IllegalStateException("Encoder output format changed twice.")
                    }

                    val outputFormat = encoder.outputFormat
                    muxerState[0] = muxer.addTrack(outputFormat)
                    muxer.start()
                    muxerState[1] = 1

                    Log.i("POTHOLE_AI", "MP4 muxer started.")
                }

                outputIndex >= 0 -> {
                    attempts = 0
                    val outputBuffer = encoder.getOutputBuffer(outputIndex)
                        ?: throw IllegalStateException("Encoder output buffer is null.")

                    if ((bufferInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) != 0) {
                        bufferInfo.size = 0
                    }

                    if (bufferInfo.size > 0) {
                        if (muxerState[1] != 1) {
                            throw IllegalStateException("Muxer has not started.")
                        }

                        outputBuffer.position(bufferInfo.offset)
                        outputBuffer.limit(bufferInfo.offset + bufferInfo.size)

                        muxer.writeSampleData(
                            muxerState[0],
                            outputBuffer,
                            bufferInfo
                        )
                    }

                    val isEndOfStream =
                        (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0

                    encoder.releaseOutputBuffer(outputIndex, false)

                    if (isEndOfStream) return
                }
            }
        }
    }



    private fun finishEncoder(
        encoder: MediaCodec,
        muxer: MediaMuxer,
        bufferInfo: MediaCodec.BufferInfo,
        muxerState: IntArray,
        presentationTimeUs: Long
    ) {
        var inputIndex = -1
        var attempts = 0

        while (inputIndex < 0 && attempts < 300) {
            inputIndex = encoder.dequeueInputBuffer(10_000)

            if (inputIndex < 0) {
                drainEncoder(
                    encoder,
                    muxer,
                    bufferInfo,
                    muxerState
                )
                attempts++
            }
        }

        if (inputIndex < 0) {
            throw IllegalStateException(
                "Could not obtain encoder input buffer for finalization."
            )
        }

        encoder.queueInputBuffer(
            inputIndex,
            0,
            0,
            presentationTimeUs,
            MediaCodec.BUFFER_FLAG_END_OF_STREAM
        )

        drainEncoder(
            encoder,
            muxer,
            bufferInfo,
            muxerState,
            endOfStream = true
        )

        Log.i("POTHOLE_AI", "Video encoder finished.")
    }



    private fun createOutputVideoUri(): Uri {

        val fileName = "Pothole_Detection_${System.currentTimeMillis()}.mp4"

        val contentValues = android.content.ContentValues().apply {
            put(
                android.provider.MediaStore.MediaColumns.DISPLAY_NAME,
                fileName
            )

            put(
                android.provider.MediaStore.MediaColumns.MIME_TYPE,
                "video/mp4"
            )

            put(
                android.provider.MediaStore.MediaColumns.RELATIVE_PATH,
                android.os.Environment.DIRECTORY_MOVIES + "/PotholeDetector"
            )

            put(
                android.provider.MediaStore.MediaColumns.IS_PENDING,
                1
            )
        }

        return contentResolver.insert(
            android.provider.MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
            contentValues
        ) ?: throw IllegalStateException(
            "Unable to create output video file."
        )
    }

    private fun createVideoMuxer(
        outputUri: Uri
    ): Pair<MediaMuxer, android.os.ParcelFileDescriptor> {

        val fileDescriptor = contentResolver.openFileDescriptor(
            outputUri,
            "w"
        ) ?: throw IllegalStateException(
            "Unable to open output video file."
        )

        try {
            val muxer = MediaMuxer(
                fileDescriptor.fileDescriptor,
                MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4
            )

            return Pair(muxer, fileDescriptor)

        } catch (e: Exception) {
            fileDescriptor.close()
            throw e
        }
    }


    private fun convertImageToBitmap(image: Image): Bitmap {
        val crop = image.cropRect
        val width = crop.width()
        val height = crop.height()

        val planes = image.planes

        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val yRowStride = planes[0].rowStride
        val uRowStride = planes[1].rowStride
        val vRowStride = planes[2].rowStride

        val uPixelStride = planes[1].pixelStride
        val vPixelStride = planes[2].pixelStride

        val pixels = IntArray(width * height)

        for (row in 0 until height) {
            for (col in 0 until width) {

                val yIndex =
                    (row + crop.top) * yRowStride + col + crop.left

                val uvRow = (row + crop.top) / 2
                val uvCol = (col + crop.left) / 2

                val uIndex =
                    uvRow * uRowStride + uvCol * uPixelStride

                val vIndex =
                    uvRow * vRowStride + uvCol * vPixelStride

                val y = (yBuffer.get(yIndex).toInt() and 0xFF) - 16
                val u = (uBuffer.get(uIndex).toInt() and 0xFF) - 128
                val v = (vBuffer.get(vIndex).toInt() and 0xFF) - 128

                val yValue = maxOf(0, y)

                val red = ((298 * yValue + 409 * v + 128) shr 8)
                    .coerceIn(0, 255)

                val green = ((298 * yValue - 100 * u - 208 * v + 128) shr 8)
                    .coerceIn(0, 255)

                val blue = ((298 * yValue + 516 * u + 128) shr 8)
                    .coerceIn(0, 255)

                pixels[row * width + col] =
                    Color.rgb(red, green, blue)
            }
        }

        return Bitmap.createBitmap(
            pixels,
            width,
            height,
            Bitmap.Config.ARGB_8888
        )
    }


    private fun inspectDecoderOutput(uri: Uri) {
        var extractor: MediaExtractor? = null
        var decoder: MediaCodec? = null

        try {
            extractor = MediaExtractor()
            extractor.setDataSource(this, uri, null)

            var videoTrack = -1
            var videoFormat: MediaFormat? = null

            for (i in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: continue

                if (mime.startsWith("video/")) {
                    videoTrack = i
                    videoFormat = format
                    break
                }
            }

            if (videoTrack < 0 || videoFormat == null) {
                throw IllegalStateException("No video track found.")
            }

            extractor.selectTrack(videoTrack)

            val mime = videoFormat.getString(MediaFormat.KEY_MIME)
                ?: throw IllegalStateException("Video MIME type missing.")

            val width = videoFormat.getInteger(MediaFormat.KEY_WIDTH)
            val height = videoFormat.getInteger(MediaFormat.KEY_HEIGHT)

            decoder = createVideoDecoder(videoFormat)


            val info = MediaCodec.BufferInfo()
            var inputFinished = false
            var frameFound = false

            while (!frameFound) {
                if (!inputFinished) {
                    val inputIndex = decoder.dequeueInputBuffer(10_000)

                    if (inputIndex >= 0) {
                        val inputBuffer = decoder.getInputBuffer(inputIndex)
                            ?: throw IllegalStateException("Decoder input buffer unavailable.")

                        val sampleSize = extractor.readSampleData(inputBuffer, 0)

                        if (sampleSize < 0) {
                            decoder.queueInputBuffer(
                                inputIndex, 0, 0, 0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM
                            )
                            inputFinished = true
                        } else {
                            decoder.queueInputBuffer(
                                inputIndex,
                                0,
                                sampleSize,
                                extractor.sampleTime,
                                0
                            )
                            extractor.advance()
                        }
                    }
                }

                val outputIndex = decoder.dequeueOutputBuffer(info, 10_000)

                if (outputIndex >= 0) {
                    if (info.size > 0) {
                        val image = decoder.getOutputImage(outputIndex)

                        if (image != null) {
                            try {
                                val bitmap = convertImageToBitmap(image)

                                Log.i(
                                    "POTHOLE_AI",
                                    "DECODER TEST SUCCESS: ${bitmap.width}x${bitmap.height}"
                                )

                                bitmap.recycle()
                                frameFound = true
                            } finally {
                                image.close()
                            }
                        } else {
                            Log.e(
                                "POTHOLE_AI",
                                "Decoder returned no accessible output image."
                            )
                            throw IllegalStateException(
                                "Decoder output image is unavailable."
                            )
                        }
                    }

                    decoder.releaseOutputBuffer(outputIndex, false)

                    if ((info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                        break
                    }
                } else if (
                    outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED
                ) {
                    Log.i(
                        "POTHOLE_AI",
                        "DECODER OUTPUT FORMAT: ${decoder.outputFormat}"
                    )
                }
            }

            if (!frameFound) {
                throw IllegalStateException("No decoded video frame was obtained.")
            }

        } catch (e: Exception) {
            Log.e("POTHOLE_AI", "Decoder test failed: ${e.message}", e)
        } finally {
            try {
                decoder?.stop()
            } catch (_: Exception) {
            }

            decoder?.release()
            extractor?.release()
        }
    }


    private fun decodeVideoFrames(
        uri: Uri,
        onFrame: (Bitmap, Long) -> Unit
    ): Int {

        var extractor: MediaExtractor? = null
        var decoder: MediaCodec? = null

        var frameCount = 0
        var inputFinished = false
        var outputFinished = false
        var idleCount = 0

        try {
            extractor = MediaExtractor()
            extractor.setDataSource(this, uri, null)

            var videoTrack = -1
            var videoFormat: MediaFormat? = null

            for (i in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: continue

                if (mime.startsWith("video/")) {
                    videoTrack = i
                    videoFormat = format
                    break
                }
            }

            if (videoTrack < 0 || videoFormat == null) {
                throw IllegalStateException("No video track found.")
            }

            extractor.selectTrack(videoTrack)

            decoder = createVideoDecoder(videoFormat)

            val bufferInfo = MediaCodec.BufferInfo()

            while (!outputFinished) {

                if (!inputFinished) {
                    val inputIndex = decoder.dequeueInputBuffer(10_000)

                    if (inputIndex >= 0) {
                        val inputBuffer = decoder.getInputBuffer(inputIndex)
                            ?: throw IllegalStateException(
                                "Decoder input buffer unavailable."
                            )

                        inputBuffer.clear()

                        val sampleSize = extractor.readSampleData(
                            inputBuffer,
                            0
                        )

                        if (sampleSize < 0) {
                            decoder.queueInputBuffer(
                                inputIndex,
                                0,
                                0,
                                0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM
                            )

                            inputFinished = true

                        } else {
                            decoder.queueInputBuffer(
                                inputIndex,
                                0,
                                sampleSize,
                                extractor.sampleTime,
                                0
                            )

                            extractor.advance()
                        }
                    }
                }

                val outputIndex = decoder.dequeueOutputBuffer(
                    bufferInfo,
                    10_000
                )

                when {
                    outputIndex >= 0 -> {
                        idleCount = 0

                        try {
                            if (bufferInfo.size > 0) {
                                val image = decoder.getOutputImage(outputIndex)
                                    ?: throw IllegalStateException(
                                        "Decoded output image unavailable."
                                    )

                                val bitmap = try {
                                    convertImageToBitmap(image)
                                } finally {
                                    image.close()
                                }

                                try {
                                    onFrame(
                                        bitmap,
                                        bufferInfo.presentationTimeUs
                                    )
                                } finally {
                                    bitmap.recycle()
                                }

                                frameCount++
                            }

                            if (
                                (bufferInfo.flags and
                                        MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                            ) {
                                outputFinished = true
                            }

                        } finally {
                            decoder.releaseOutputBuffer(
                                outputIndex,
                                false
                            )
                        }
                    }

                    outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        Log.i(
                            "POTHOLE_AI",
                            "VIDEO OUTPUT FORMAT: ${decoder.outputFormat}"
                        )
                    }

                    outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                        if (inputFinished) {
                            idleCount++

                            if (idleCount > 500) {
                                throw IllegalStateException(
                                    "Timed out waiting for decoder output."
                                )
                            }
                        }
                    }
                }
            }

            Log.i(
                "POTHOLE_AI",
                "VIDEO DECODING COMPLETED. FRAMES: $frameCount"
            )

            return frameCount

        } finally {
            try {
                decoder?.stop()
            } catch (_: Exception) {
            }

            decoder?.release()
            extractor?.release()
        }
    }


    private fun testCompleteVideoDecoding(uri: Uri) {
        runOnUiThread {
            tvStatus.text = "Testing complete video decoding..."
        }

        executor.execute {
            try {
                val totalFrames = decodeVideoFrames(uri) { bitmap, timestampUs ->

                    if (timestampUs % 1_000_000L < 40_000L) {
                        Log.i(
                            "POTHOLE_AI",
                            "Decoded frame timestamp: ${timestampUs / 1000} ms"
                        )
                    }
                }

                runOnUiThread {
                    tvStatus.text =
                        "Video decoding successful: $totalFrames frames"

                    Toast.makeText(
                        this,
                        "Decoded $totalFrames video frames successfully.",
                        Toast.LENGTH_LONG
                    ).show()
                }

            } catch (e: Exception) {
                Log.e(
                    "POTHOLE_AI",
                    "COMPLETE VIDEO DECODING FAILED",
                    e
                )

                runOnUiThread {
                    tvStatus.text = "Video decoding failed: ${e.message}"
                }
            }
        }
    }




    private fun inspectEncoderInputLayout() {

        var encoder: MediaCodec? = null

        try {
            Log.i("POTHOLE_AI", "Starting encoder input inspection...")

            encoder = createVideoEncoder(
                1280,
                720,
                OUTPUT_VIDEO_FPS
            )

            encoder.start()

            val inputIndex = encoder.dequeueInputBuffer(1_000_000)

            if (inputIndex < 0) {
                throw IllegalStateException(
                    "Encoder input buffer was not available."
                )
            }

            val inputBuffer = encoder.getInputBuffer(inputIndex)
                ?: throw IllegalStateException(
                    "Unable to access encoder input buffer."
                )

            val inputFormat = encoder.inputFormat

            val stride = if (inputFormat.containsKey(MediaFormat.KEY_STRIDE)) {
                inputFormat.getInteger(MediaFormat.KEY_STRIDE)
            } else {
                1280
            }

            val sliceHeight = if (
                inputFormat.containsKey(MediaFormat.KEY_SLICE_HEIGHT)
            ) {
                inputFormat.getInteger(MediaFormat.KEY_SLICE_HEIGHT)
            } else {
                720
            }

            Log.i(
                "POTHOLE_AI",
                "ENCODER INPUT FORMAT: $inputFormat"
            )

            Log.i(
                "POTHOLE_AI",
                "ENCODER INPUT BUFFER CAPACITY: ${inputBuffer.capacity()}"
            )

            Log.i(
                "POTHOLE_AI",
                "ENCODER STRIDE: $stride"
            )

            Log.i(
                "POTHOLE_AI",
                "ENCODER SLICE HEIGHT: $sliceHeight"
            )

            Log.i(
                "POTHOLE_AI",
                "EXPECTED FRAME SIZE: ${stride * sliceHeight * 3 / 2}"
            )

            Log.i(
                "POTHOLE_AI",
                "Encoder input inspection completed."
            )

        } catch (e: Exception) {

            Log.e(
                "POTHOLE_AI",
                "Encoder input inspection failed",
                e
            )

        } finally {

            try {
                encoder?.stop()
            } catch (_: Exception) {
            }

            try {
                encoder?.release()
            } catch (_: Exception) {
            }
        }
    }

    private fun analyzeCameraFrame(imageProxy: ImageProxy) {

        try {
            val currentTime = System.currentTimeMillis()

            // Limit inference to approximately 5 FPS.
            if (currentTime - lastInferenceTime < 200) {
                return
            }

            lastInferenceTime = currentTime

            val bitmap = imageProxyToBitmap(imageProxy)

            if (bitmap != null) {
                try {
                    val detections = engine?.detect(bitmap) ?: emptyList()

                    val frameWidth = bitmap.width
                    val frameHeight = bitmap.height

                    // Record pothole detection without GPS.
                    if (detections.isNotEmpty() &&
                        currentTime - lastPotholeRecordTime >= potholeRecordCooldown
                    ) {

                        val bestDetection = detections.maxByOrNull {
                            it.confidence
                        }

                        if (bestDetection != null) {

                            val record = PotholeRecord(
                                confidence = bestDetection.confidence,
                                timestamp = currentTime
                            )

                            detectionRecords.add(record)
                            savePotholeRecord(record)

                            latestDetectionText = "Pothole detected by live camera"
                            latestConfidenceText = "Confidence: ${
                                String.format(
                                    Locale.US,
                                    "%.2f",
                                    bestDetection.confidence * 100
                                )
                            }%"

                            lastPotholeRecordTime = currentTime
                        }
                    }

                    runOnUiThread {

                        detectionOverlay.updateDetections(
                            detections,
                            frameWidth,
                            frameHeight
                        )

                        tvStatus.text =
                            "Status: ${detections.size} pothole(s) detected | " +
                                    "Records: ${detectionRecords.size}"
                    }

                    Log.i(
                        "POTHOLE_AI",
                        "Live camera detections: ${detections.size}"
                    )

                } finally {
                    bitmap.recycle()
                }
            }

        } catch (e: Exception) {

            Log.e("POTHOLE_AI", "Camera analysis error", e)

        } finally {
            imageProxy.close()
        }
    }




    private fun showPotholeHistory() {

        if (detectionRecords.isEmpty()) {
            Toast.makeText(
                this,
                "No pothole records available yet.",
                Toast.LENGTH_SHORT
            ).show()
            return
        }

        val historyText = StringBuilder()

        detectionRecords.forEachIndexed { index, record ->

            val date = java.text.SimpleDateFormat(
                "dd/MM/yyyy HH:mm:ss",
                java.util.Locale.getDefault()
            ).format(java.util.Date(record.timestamp))

            historyText.append(
                """
            Detection ${index + 1}
            Confidence: ${(record.confidence * 100).toInt()}%
            Date: $date

            -------------------------

            """.trimIndent()
            )
        }

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Pothole Detection History")
            .setMessage(historyText.toString())
            .setPositiveButton("CLOSE", null)
            .setNegativeButton("CLEAR HISTORY") { _, _ ->

                androidx.appcompat.app.AlertDialog.Builder(this)
                    .setTitle("Confirm Deletion")
                    .setMessage(
                        "Are you sure you want to delete all detection history? This cannot be undone."
                    )
                    .setNegativeButton("CANCEL", null)
                    .setPositiveButton("DELETE") { _, _ ->

                        detectionRecords.clear()

                        getSharedPreferences(
                            "PotholeHistory",
                            MODE_PRIVATE
                        ).edit()
                            .remove("records")
                            .apply()

                        findViewById<TextView>(
                            R.id.tvHomeTotalDetections
                        )?.text = "0"

                        Toast.makeText(
                            this,
                            "Detection history cleared.",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                    .show()
            }
            .show()
    }


    private fun savePotholeRecord(record: PotholeRecord) {

        val preferences = getSharedPreferences(
            "PotholeHistory",
            MODE_PRIVATE
        )

        val recordsJson = preferences.getString(
            "records",
            "[]"
        )

        val recordsArray = JSONArray(recordsJson)

        val jsonObject = JSONObject().apply {
            put("confidence", record.confidence)
            put("timestamp", record.timestamp)
        }

        recordsArray.put(jsonObject)

        preferences.edit()
            .putString("records", recordsArray.toString())
            .apply()
    }

    private fun loadPotholeRecords() {

        val preferences = getSharedPreferences(
            "PotholeHistory",
            MODE_PRIVATE
        )

        val recordsJson = preferences.getString(
            "records",
            "[]"
        )

        try {
            val recordsArray = JSONArray(recordsJson)

            detectionRecords.clear()

            for (i in 0 until recordsArray.length()) {

                val jsonObject = recordsArray.getJSONObject(i)

                val record = PotholeRecord(
                    confidence = jsonObject.getDouble("confidence").toFloat(),
                    timestamp = jsonObject.getLong("timestamp")
                )

                detectionRecords.add(record)
            }

            Log.i(
                "POTHOLE_AI",
                "Loaded ${detectionRecords.size} saved pothole records"
            )

        } catch (e: Exception) {
            Log.e("POTHOLE_AI", "Error loading pothole records", e)
        }
    }


    private fun showHomeDashboard() {
        setContentView(R.layout.activity_home)
        val homeScrollView = findViewById<ScrollView>(R.id.homeScrollView)

        homeScrollView.post {
            Log.d(
                "DASHBOARD_TEST",
                "Scroll Y: ${homeScrollView.scrollY}, Height: ${homeScrollView.height}, Content Height: ${homeScrollView.getChildAt(0).height}"
            )
        }
        val tvHomeTotalDetections =
            findViewById<TextView>(R.id.tvHomeTotalDetections)

        val btnMenu = findViewById<TextView>(R.id.btnMenu)

        btnMenu.setOnClickListener {
            val popupMenu = PopupMenu(this, btnMenu)

            popupMenu.menu.add("About")
            popupMenu.menu.add("Exit")

            popupMenu.setOnMenuItemClickListener { item ->

                when (item.title.toString()) {

                    "About" -> {
                        showAboutDialog()
                        true
                    }

                    "Exit" -> {
                        finishAffinity()
                        true
                    }

                    else -> false
                }
            }

            popupMenu.show()
        }
        val btnHomeLive = findViewById<Button>(R.id.btnHomeLive)
        val btnHomeImage = findViewById<Button>(R.id.btnHomeImage)
        val btnHomeVideo = findViewById<Button>(R.id.btnHomeVideo)
        val btnHomeHistory = findViewById<Button>(R.id.btnHomeHistory)

        tvHomeTotalDetections.text = detectionRecords.size.toString()
        findViewById<TextView>(R.id.tvHomeLatestResult).text =
            latestDetectionText

        findViewById<TextView>(R.id.tvHomeConfidence).text =
            latestConfidenceText

        btnHomeLive.setOnClickListener {
            showDetectionWorkspace("live")
            requestCameraPermissionAndStart()
        }

        btnHomeImage.setOnClickListener {
            showDetectionWorkspace("image")
        }

        btnHomeVideo.setOnClickListener {
            showDetectionWorkspace("video")
        }

        btnHomeHistory.setOnClickListener {
            showPotholeHistory()
        }
    }


    private fun showAboutDialog() {

        val aboutMessage = """
        AI POTHOLE DETECTOR

        Developer:
        Sani Hassan Muhammad (P25EGCP8021)

        TECHNOLOGY

        AI Model: YOLO26n
        Inference Engine: ONNX Runtime
        Platform: Android

        FEATURES

        • Image-based pothole detection
        • Video-based pothole detection
        • Real-time camera detection
        • Detection history

        ABOUT

        An AI-powered Android application
        designed to detect potholes and
        support road-defect monitoring
        using computer vision and deep learning.

        MSc Artificial Intelligence 
        COEN872 Project-Based Examination
        2025/2026 Session
    """.trimIndent()

        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("About the Developer & App")
            .setMessage(aboutMessage)
            .setPositiveButton("CLOSE", null)
            .show()
    }


    private fun showDetectionWorkspace(mode: String) {
        setContentView(R.layout.activity_detection)

        val btnMenu = findViewById<TextView>(R.id.btnMenu)

        btnMenu.setOnClickListener {
            val popupMenu = PopupMenu(this, btnMenu)

            popupMenu.menu.add("About")
            popupMenu.menu.add("Exit")

            popupMenu.setOnMenuItemClickListener { item ->

                when (item.title.toString()) {

                    "About" -> {
                        showAboutDialog()
                        true
                    }

                    "Exit" -> {
                        finishAffinity()
                        true
                    }

                    else -> false
                }
            }

            popupMenu.show()
        }

        // Image detection controls
        findViewById<Button>(R.id.btnSelectImage).visibility =
            if (mode == "image") View.VISIBLE else View.GONE

        findViewById<Button>(R.id.btnDetectImage).visibility =
            if (mode == "image") View.VISIBLE else View.GONE

        findViewById<Button>(R.id.btnSaveDetection).visibility =
            if (mode == "image") View.VISIBLE else View.GONE

// Video detection controls
        findViewById<Button>(R.id.btnSelectVideo).visibility =
            if (mode == "video") View.VISIBLE else View.GONE

        findViewById<Button>(R.id.btnPlayAnnotatedVideo).visibility =
            if (mode == "video") View.VISIBLE else View.GONE

        findViewById<Button>(R.id.btnSaveAnnotatedVideo).visibility =
            if (mode == "video") View.VISIBLE else View.GONE

// Real-time camera control
        findViewById<Button>(R.id.btnRealtimeCamera).visibility =
            if (mode == "live") View.VISIBLE else View.GONE

// History is accessed from the Home Dashboard
        findViewById<Button>(R.id.btnViewHistory).visibility = View.GONE
        findViewById<Button>(R.id.btnBackHome).setOnClickListener {
            showHomeDashboard()
        }

        cameraPreview = findViewById(R.id.cameraPreview)
        detectionOverlay = findViewById(R.id.detectionOverlay)
        videoPreview = findViewById(R.id.videoPreview)
        // Show only the preview for the selected mode
        findViewById<ImageView>(R.id.imagePreview).visibility =
            if (mode == "image") View.VISIBLE else View.GONE

        findViewById<VideoView>(R.id.videoPreview).visibility =
            if (mode == "video") View.VISIBLE else View.GONE

        cameraPreview.visibility =
            if (mode == "live") View.VISIBLE else View.GONE

        detectionOverlay.visibility =
            if (mode == "live") View.VISIBLE else View.GONE

        findViewById<TextView>(R.id.tvImagePlaceholder).visibility =
            if (mode == "image") View.VISIBLE else View.GONE

        initializeViews()
        setupButtons()

        btnViewHistory.setOnClickListener {
            showPotholeHistory()
        }

        btnRealtimeCamera = findViewById(R.id.btnRealtimeCamera)

        btnRealtimeCamera.setOnClickListener {
            requestCameraPermissionAndStart()
        }

        checkVideoEncoder()

        btnPlayAnnotatedVideo =
            findViewById(R.id.btnPlayAnnotatedVideo)

        btnSaveAnnotatedVideo =
            findViewById(R.id.btnSaveAnnotatedVideo)
    }


}
