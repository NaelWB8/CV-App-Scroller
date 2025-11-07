package com.palma.gesture

import android.graphics.PointF
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.*

data class EngineOut(val state: String, val cursor: PointF, val events: List<String>)

class GestureEngine {

    // Thresholds for gesture recognition
    private val angleExt = 160.0
    private val angleFold = 130.0
    private val towardCamZ = -0.03f
    private val thumbSepMin = 0.30

    // Sensitivity and smoothing (tune these)
    private val gainX = 2.2f   // increased gain for livelier response
    private val gainY = 2.0f
    private val smooth = 0.0f
    private val minStable = 2

    // Cursor / state
    private var cx = 0.5f
    private var cy = 0.5f
    private var lastIx = 0f
    private var lastIy = 0f
    private var haveLast = false
    private var state = "IDLE"
    private var pending: String? = null
    private var pendingCount = 0

    // Smoothing
    private var smoothedX = 0.5f
    private var smoothedY = 0.5f
    private val alpha = 0.65f // higher = smoother but slower
    private val deadzone = 0.01f // ignore micro-movement jitter

    // safety clamps
    private val maxDelta = 0.35f // if dx/dy bigger than this, treat as glitch

    fun process(res: HandLandmarkerResult?): EngineOut {
        var newState = "IDLE"
        val events = mutableListOf<String>()
        var outX = cx
        var outY = cy

        val hand = res?.landmarks()?.firstOrNull()
        if (hand != null) {
            fun p(i: Int) = hand[i]
            val pointing = isIndexPointing(p(5), p(6), p(7), p(8))
            val thumbOut = isThumbOut(p(2), p(3), p(4), p(8), p(0), p(9))
            val thumbFold = isThumbFolded(p(2), p(3), p(4))

            newState = when {
                pointing && thumbFold -> "HOLD"
                pointing && thumbOut -> "POINTER"
                else -> "IDLE"
            }

            if (newState == "POINTER" || newState == "HOLD") {
                // IMPORTANT: use raw normalized landmark as-is (0..1, origin = left/top of image)
                val ix = p(8).x()
                val iy = p(8).y()

                // sanity check: ignore bad coords
                if (ix.isNaN() || iy.isNaN() || ix < -0.2f || ix > 1.2f || iy < -0.2f || iy > 1.2f) {
                    // don't update haveLast; treat as transient miss
                } else {
                    if (!haveLast) {
                        // anchor positions on first good detection to avoid jump-to-edge
                        lastIx = ix
                        lastIy = iy
                        cx = ix.coerceIn(0f, 1f)
                        cy = iy.coerceIn(0f, 1f)
                        smoothedX = cx
                        smoothedY = cy
                        haveLast = true
                    }

                    // compute deltas
                    val rawDx = (ix - lastIx) * gainX
                    val rawDy = (iy - lastIy) * gainY
                    lastIx = ix
                    lastIy = iy

                    // clamp improbable spikes (caused by temporary bad detection)
                    val dx = when {
                        rawDx.isNaN() -> 0f
                        abs(rawDx) > maxDelta -> 0f
                        abs(rawDx) > deadzone -> rawDx
                        else -> 0f
                    }
                    val dy = when {
                        rawDy.isNaN() -> 0f
                        abs(rawDy) > maxDelta -> 0f
                        abs(rawDy) > deadzone -> rawDy
                        else -> 0f
                    }

                    // target position (delta-based movement)
                    val targetX = (cx + dx).coerceIn(0f, 1f)
                    val targetY = (cy + dy).coerceIn(0f, 1f)

                    // exponential smoothing
                    smoothedX = alpha * smoothedX + (1 - alpha) * targetX
                    smoothedY = alpha * smoothedY + (1 - alpha) * targetY

                    cx = cx * smooth + smoothedX * (1 - smooth)
                    cy = cy * smooth + smoothedY * (1 - smooth)

                    outX = cx
                    outY = cy
                }
            } else {
                haveLast = false
            }
        } else haveLast = false

        // stabilize state transitions
        if (newState == state) {
            pending = null
            pendingCount = 0
        } else {
            if (pending == newState) pendingCount++ else {
                pending = newState
                pendingCount = 1
            }
            if (pendingCount >= minStable) {
                if (state != "HOLD" && newState == "HOLD") events += "HOLD_START"
                if (state == "HOLD" && newState != "HOLD") events += "HOLD_END"
                state = newState
                pending = null
                pendingCount = 0
            }
        }

        return EngineOut(state, PointF(outX, outY), events)
    }

    // ========== HAND GESTURE HELPERS ==========

    private fun isIndexPointing(
        iMcp: NormalizedLandmark, iPip: NormalizedLandmark,
        iDip: NormalizedLandmark, iTip: NormalizedLandmark
    ): Boolean {
        val straight = angle(iMcp, iPip, iDip) >= angleExt
        val toward = (iTip.z() - iPip.z()) <= towardCamZ
        return straight && toward
    }

    private fun isThumbOut(
        tMcp: NormalizedLandmark, tIp: NormalizedLandmark, tTip: NormalizedLandmark,
        idxTip: NormalizedLandmark, wrist: NormalizedLandmark, midMcp: NormalizedLandmark
    ): Boolean {
        val straight = angle(tMcp, tIp, tTip) >= angleExt
        val handUnit = dist(wrist, midMcp).coerceAtLeast(1e-6f)
        val sep = dist(tTip, idxTip) / handUnit
        return straight && sep >= thumbSepMin
    }

    private fun isThumbFolded(
        tMcp: NormalizedLandmark, tIp: NormalizedLandmark, tTip: NormalizedLandmark
    ): Boolean = angle(tMcp, tIp, tTip) <= angleFold

    private fun angle(a: NormalizedLandmark, b: NormalizedLandmark, c: NormalizedLandmark): Double {
        val v1x = a.x() - b.x()
        val v1y = a.y() - b.y()
        val v1z = a.z() - b.z()
        val v2x = c.x() - b.x()
        val v2y = c.y() - b.y()
        val v2z = c.z() - b.z()
        val num = (v1x * v2x + v1y * v2y + v1z * v2z).toDouble()
        val d1 = sqrt((v1x * v1x + v1y * v1y + v1z * v1z).toDouble())
        val d2 = sqrt((v2x * v2x + v2y * v2y + v2z * v2z).toDouble())
        val cos = (num / (d1 * d2)).coerceIn(-1.0, 1.0)
        return Math.toDegrees(acos(cos))
    }

    private fun dist(a: NormalizedLandmark, b: NormalizedLandmark): Float {
        val dx = a.x() - b.x()
        val dy = a.y() - b.y()
        val dz = a.z() - b.z()
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
}
