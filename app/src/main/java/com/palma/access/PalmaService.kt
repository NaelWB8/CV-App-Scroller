package com.palma.access

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Path
import android.view.accessibility.AccessibilityEvent

class PalmaService : AccessibilityService() {
    companion object { @Volatile var instance: PalmaService? = null }

    private var isDown = false
    private var lastX = 0f; private var lastY = 0f

    override fun onServiceConnected() { instance = this }
    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}
    override fun onInterrupt() {}
    override fun onDestroy() { instance = null; super.onDestroy() }

    fun downAt(x: Float, y: Float) {
        if (isDown) return
        isDown = true; lastX = x; lastY = y
        val path = Path().apply { moveTo(x, y) }
        val stroke = GestureDescription.StrokeDescription(path, 0, 50, true)
        dispatchGesture(GestureDescription.Builder().addStroke(stroke).build(), null, null)
    }

    fun moveTo(x: Float, y: Float) {
        if (!isDown) return
        val path = Path().apply { moveTo(lastX, lastY); lineTo(x, y) }
        val stroke = GestureDescription.StrokeDescription(path, 0, 100, true)
        dispatchGesture(GestureDescription.Builder().addStroke(stroke).build(), null, null)
        lastX = x; lastY = y
    }

    fun up() {
        if (!isDown) return
        val path = Path().apply { moveTo(lastX, lastY) }
        val stroke = GestureDescription.StrokeDescription(path, 0, 50, false)
        dispatchGesture(GestureDescription.Builder().addStroke(stroke).build(), null, null)
        isDown = false
    }
}
