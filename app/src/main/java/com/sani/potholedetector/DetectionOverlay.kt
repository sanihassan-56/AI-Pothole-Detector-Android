
package com.sani.potholedetector

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.view.View

class DetectionOverlay @JvmOverloads constructor(
    context: Context,
    attrs: android.util.AttributeSet? = null
) : View(context, attrs) {

    private val boxPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.RED
        textSize = 38f
        style = Paint.Style.FILL
        isAntiAlias = true
        isFakeBoldText = true
    }

    private var detections: List<Detection> = emptyList()
    private var sourceWidth = 1f
    private var sourceHeight = 1f

    fun updateDetections(
        newDetections: List<Detection>,
        imageWidth: Int,
        imageHeight: Int
    ) {
        detections = newDetections
        sourceWidth = imageWidth.toFloat()
        sourceHeight = imageHeight.toFloat()
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val scaleX = width / sourceWidth
        val scaleY = height / sourceHeight

        for (detection in detections) {

            val left = detection.left * scaleX
            val top = detection.top * scaleY
            val right = detection.right * scaleX
            val bottom = detection.bottom * scaleY

            canvas.drawRect(
                left, top, right, bottom, boxPaint
            )

            val label = "POTHOLE ${
                (detection.confidence * 100).toInt()
            }%"

            canvas.drawText(
                label,
                left,
                (top - 12f).coerceAtLeast(38f),
                textPaint
            )
        }
    }
}
