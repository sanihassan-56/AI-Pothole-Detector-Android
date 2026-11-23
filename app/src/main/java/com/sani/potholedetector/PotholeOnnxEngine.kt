
package com.sani.potholedetector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

data class Detection(
    val classId: Int,
    val className: String,
    val confidence: Float,
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
)

class PotholeOnnxEngine(context: Context) : Closeable {

    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String

    companion object {
        private const val MODEL_NAME = "best.onnx"
        private const val INPUT_SIZE = 640
        private const val CONFIDENCE_THRESHOLD = 0.25f
        private const val IOU_THRESHOLD = 0.45f

        private val CLASS_NAMES = arrayOf(
            "cracks",
            "patching",
            "pothole"
        )
    }

    init {
        val modelBytes = context.applicationContext.assets
            .open(MODEL_NAME)
            .use { it.readBytes() }

        val options = OrtSession.SessionOptions()
        try {
            session = environment.createSession(modelBytes, options)
        } finally {
            options.close()
        }

        inputName = session.inputNames.first()
    }

    fun getModelInformation(): String {
        val information = StringBuilder()

        information.append("YOLO26n Model Loaded Successfully\n")

        information.append("\nINPUT DETAILS\n")
        for ((name, info) in session.inputInfo) {
            information.append("Name: $name\n")
            information.append("Details: ${info.info}\n")
        }

        information.append("\nOUTPUT DETAILS\n")
        for ((name, info) in session.outputInfo) {
            information.append("Name: $name\n")
            information.append("Details: ${info.info}\n")
        }

        return information.toString()
    }

    // Convert the image into the model's RGB input tensor.
    private fun preprocess(bitmap: Bitmap): Pair<FloatArray, FloatArray> {

        val originalWidth = bitmap.width.toFloat()
        val originalHeight = bitmap.height.toFloat()

        val scale = min(
            INPUT_SIZE / originalWidth,
            INPUT_SIZE / originalHeight
        )

        val resizedWidth = originalWidth * scale
        val resizedHeight = originalHeight * scale

        val padX = (INPUT_SIZE - resizedWidth) / 2f
        val padY = (INPUT_SIZE - resizedHeight) / 2f

        val resizedBitmap = Bitmap.createBitmap(
            INPUT_SIZE,
            INPUT_SIZE,
            Bitmap.Config.ARGB_8888
        )

        val canvas = Canvas(resizedBitmap)
        canvas.drawColor(Color.rgb(114, 114, 114))

        val paint = Paint(Paint.FILTER_BITMAP_FLAG)

        canvas.drawBitmap(
            bitmap,
            null,
            android.graphics.RectF(
                padX,
                padY,
                padX + resizedWidth,
                padY + resizedHeight
            ),
            paint
        )

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)

        resizedBitmap.getPixels(
            pixels,
            0,
            INPUT_SIZE,
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
        )

        val inputData = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
        val planeSize = INPUT_SIZE * INPUT_SIZE

        for (i in pixels.indices) {
            val pixel = pixels[i]

            inputData[i] =
                Color.red(pixel) / 255f

            inputData[planeSize + i] =
                Color.green(pixel) / 255f

            inputData[2 * planeSize + i] =
                Color.blue(pixel) / 255f
        }

        resizedBitmap.recycle()

        return Pair(
            inputData,
            floatArrayOf(scale, padX, padY)
        )
    }

    // Run real image inference.
    fun detect(bitmap: Bitmap): List<Detection> {

        require(!bitmap.isRecycled) {
            "Input bitmap has been recycled"
        }

        val (inputData, transform) = preprocess(bitmap)

        val scale = transform[0]
        val padX = transform[1]
        val padY = transform[2]

        val tensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(inputData),
            longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
        )

        try {
            val results = session.run(
                mapOf(inputName to tensor)
            )

            try {
                val output = results[0].value as Array<*>
                val channels = output[0] as Array<*>

                require(channels.size == 7) {
                    "Unexpected model output: expected 7 channels, got ${channels.size}"
                }

                val xCenter = channels[0] as FloatArray
                val yCenter = channels[1] as FloatArray
                val widths = channels[2] as FloatArray
                val heights = channels[3] as FloatArray

                val classScores = Array(3) {
                    channels[it + 4] as FloatArray
                }

                val candidates = mutableListOf<Detection>()

                for (i in xCenter.indices) {

                    var bestClass = -1
                    var bestConfidence = 0f

                    for (classId in CLASS_NAMES.indices) {
                        val confidence = classScores[classId][i]

                        if (confidence > bestConfidence) {
                            bestConfidence = confidence
                            bestClass = classId
                        }
                    }

                    if (bestConfidence < CONFIDENCE_THRESHOLD) {
                        continue
                    }

                    val x1 = (xCenter[i] - widths[i] / 2f - padX) / scale
                    val y1 = (yCenter[i] - heights[i] / 2f - padY) / scale
                    val x2 = (xCenter[i] + widths[i] / 2f - padX) / scale
                    val y2 = (yCenter[i] + heights[i] / 2f - padY) / scale

                    val left = x1.coerceIn(0f, bitmap.width.toFloat())
                    val top = y1.coerceIn(0f, bitmap.height.toFloat())
                    val right = x2.coerceIn(0f, bitmap.width.toFloat())
                    val bottom = y2.coerceIn(0f, bitmap.height.toFloat())

                    if (right <= left || bottom <= top) {
                        continue
                    }

                    candidates.add(
                        Detection(
                            classId = bestClass,
                            className = CLASS_NAMES[bestClass],
                            confidence = bestConfidence,
                            left = left,
                            top = top,
                            right = right,
                            bottom = bottom
                        )
                    )
                }

                return applyNms(candidates).filter {
                    it.classId == 2
                }

            } finally {
                results.close()
            }

        } finally {
            tensor.close()
        }
    }

    // Intersection over Union.
    private fun calculateIoU(a: Detection, b: Detection): Float {

        val intersectionLeft = max(a.left, b.left)
        val intersectionTop = max(a.top, b.top)
        val intersectionRight = min(a.right, b.right)
        val intersectionBottom = min(a.bottom, b.bottom)

        val intersectionWidth = max(
            0f,
            intersectionRight - intersectionLeft
        )

        val intersectionHeight = max(
            0f,
            intersectionBottom - intersectionTop
        )

        val intersection = intersectionWidth * intersectionHeight

        val areaA =
            (a.right - a.left) * (a.bottom - a.top)

        val areaB =
            (b.right - b.left) * (b.bottom - b.top)

        return intersection / (areaA + areaB - intersection + 1e-6f)
    }

    // Class-aware Non-Maximum Suppression.
    private fun applyNms(
        detections: List<Detection>
    ): List<Detection> {

        val selected = mutableListOf<Detection>()

        for (classId in CLASS_NAMES.indices) {

            val remaining = detections
                .filter { it.classId == classId }
                .sortedByDescending { it.confidence }
                .toMutableList()

            while (remaining.isNotEmpty()) {

                val best = remaining.removeAt(0)
                selected.add(best)

                val iterator = remaining.iterator()

                while (iterator.hasNext()) {
                    val candidate = iterator.next()

                    if (calculateIoU(best, candidate) > IOU_THRESHOLD) {
                        iterator.remove()
                    }
                }
            }
        }

        return selected.sortedByDescending { it.confidence }
    }

    override fun close() {
        session.close()
    }
}
