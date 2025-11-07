package com.palma.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class CursorOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val paint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private var cursorX = -1f
    private var cursorY = -1f

    fun updateCursor(x: Float, y: Float) {
        cursorX = x
        cursorY = y
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (cursorX >= 0 && cursorY >= 0) {
            canvas.drawCircle(cursorX, cursorY, 25f, paint)
        }
    }
}
