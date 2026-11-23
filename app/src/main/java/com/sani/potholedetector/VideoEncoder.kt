package com.sani.potholedetector

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.view.Surface
import java.io.File

class VideoEncoder(
    private val outputFile: File,
    private val width: Int,
    private val height: Int,
    private val frameRate: Int = 25
) {

    // Encoder implementation will be added in the next step.
    private var encoder: MediaCodec? = null
    private var inputSurface: Surface? = null
    private var muxer: MediaMuxer? = null
    private var renderer: EglFrameRenderer? = null

    private var videoTrackIndex = -1
    private var muxerStarted = false
    private var isFinished = false

    private val bufferInfo = MediaCodec.BufferInfo()

    private val presentationTimeStepUs = 1_000_000L / frameRate
    private var frameCount = 0L

    fun start() {
        require(width > 0 && height > 0) {
            "Video dimensions must be greater than zero"
        }

        val format = MediaFormat.createVideoFormat(
            MediaFormat.MIMETYPE_VIDEO_AVC,
            width,
            height
        )

        format.setInteger(
            MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatSurface
        )

        format.setInteger(MediaFormat.KEY_BIT_RATE, 4_000_000)
        format.setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
        format.setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2)

        encoder = MediaCodec.createEncoderByType(
            MediaFormat.MIMETYPE_VIDEO_AVC
        )

        encoder!!.configure(
            format,
            null,
            null,
            MediaCodec.CONFIGURE_FLAG_ENCODE
        )

        inputSurface = encoder!!.createInputSurface()
        encoder!!.start()

        muxer = MediaMuxer(
            outputFile.absolutePath,
            MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4
        )
        renderer = EglFrameRenderer(
            inputSurface!!,
            width,
            height
        )

        renderer!!.initialize()
    }

    fun writeFrame(bitmap: Bitmap, presentationTimeUs: Long) {
        check(!isFinished) {
            "Encoder has already been finished"
        }

        renderer?.drawFrame(bitmap, presentationTimeUs)
            ?: throw IllegalStateException("Renderer is not initialized")

        drainEncoder()
    }

    fun drainEncoder(endOfStream: Boolean = false) {
        val codec = encoder ?: return

        if (endOfStream) {
            codec.signalEndOfInputStream()
        }

        var encoderDone = false

        while (!encoderDone) {
            val outputIndex = codec.dequeueOutputBuffer(
                bufferInfo,
                10_000
            )

            when {
                outputIndex == MediaCodec.INFO_TRY_AGAIN_LATER -> {
                    if (!endOfStream) {
                        return
                    }
                }

                outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    if (muxerStarted) {
                        throw IllegalStateException(
                            "Encoder output format changed twice"
                        )
                    }

                    val newFormat = codec.outputFormat
                    videoTrackIndex = muxer!!.addTrack(newFormat)
                    muxer!!.start()
                    muxerStarted = true
                }

                outputIndex >= 0 -> {
                    val encodedData = codec.getOutputBuffer(outputIndex)

                    if (bufferInfo.flags and
                        MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0
                    ) {
                        bufferInfo.size = 0
                    }

                    if (bufferInfo.size > 0) {
                        if (!muxerStarted) {
                            throw IllegalStateException(
                                "Muxer has not started"
                            )
                        }

                        encodedData?.position(bufferInfo.offset)
                        encodedData?.limit(
                            bufferInfo.offset + bufferInfo.size
                        )

                        if (encodedData != null) {
                            muxer!!.writeSampleData(
                                videoTrackIndex,
                                encodedData,
                                bufferInfo
                            )
                        }
                    }

                    if (bufferInfo.flags and
                        MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    ) {
                        encoderDone = true
                    }

                    codec.releaseOutputBuffer(outputIndex, false)
                }
            }
        }
    }

    fun finish() {
        if (isFinished) return

        try {
            drainEncoder(endOfStream = true)
        } finally {
            renderer?.release()
            renderer = null

            release()
            isFinished = true
        }
    }

    fun release() {
        try {
            renderer?.release()
        } catch (e: Exception) {
            e.printStackTrace()
        }
        renderer = null
        try {
            encoder?.stop()
        } catch (e: Exception) {
            e.printStackTrace()
        }

        try {
            encoder?.release()
        } catch (e: Exception) {
            e.printStackTrace()
        }

        try {
            inputSurface?.release()
        } catch (e: Exception) {
            e.printStackTrace()
        }

        try {
            if (muxerStarted) {
                muxer?.stop()
            }
            muxer?.release()
        } catch (e: Exception) {
            e.printStackTrace()
        }

        encoder = null
        inputSurface = null
        muxer = null
        muxerStarted = false
    }
}