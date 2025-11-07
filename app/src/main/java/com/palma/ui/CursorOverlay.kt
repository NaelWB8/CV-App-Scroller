package com.palma.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class CursorOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var cursorX = 0f
    private var cursorY = 0f
    private val cursorPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
    }

    fun updateCursor(x: Float, y: Float) {
        cursorX = x
        cursorY = y
        invalidate() // Redraw the view
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawCircle(cursorX, cursorY, 20f, cursorPaint)
    }
}
