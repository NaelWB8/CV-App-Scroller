package com.palma.gesture

import android.graphics.PointF
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.*

data class EngineOut(val state: String, val cursor: PointF, val events: List<String>)

class GestureEngine {
    private val angleExt = 160.0
    private val angleFold = 130.0
    private val towardCamZ = -0.03f
    private val thumbSepMin = 0.30
    private val gainX = 1.6f
    private val gainY = 1.6f
    private val smooth = 0.35f
    private val minStable = 2

    private var cx = 0.5f; private var cy = 0.5f
    private var lastIx = 0f; private var lastIy = 0f; private var haveLast = false
    private var state = "IDLE"; private var pending: String? = null; private var pendingCount = 0

    fun process(res: HandLandmarkerResult?): EngineOut {
        var newState = "IDLE"
        val events = mutableListOf<String>()
        var outX = cx; var outY = cy

        val hand = res?.landmarks()?.firstOrNull()
        if (hand != null) {
            fun p(i: Int) = hand[i]
            val pointing = isIndexPointing(p(5), p(6), p(7), p(8))
            val thumbOut = isThumbOut(p(2), p(3), p(4), p(8), p(0), p(9))
            val thumbFold = isThumbFolded(p(2), p(3), p(4))

            newState = when {
                pointing && thumbFold -> "HOLD"
                pointing && thumbOut  -> "POINTER"
                else                  -> "IDLE"
            }

            if (newState == "POINTER" || newState == "HOLD") {
                val ix = 1f - p(8).x()
                val iy = p(8).y()
                if (!haveLast) { lastIx = ix; lastIy = iy; haveLast = true }
                val dx = (ix - lastIx) * gainX
                val dy = (iy - lastIy) * gainY
                lastIx = ix; lastIy = iy
                val tx = (cx + dx).coerceIn(0f, 1f)
                val ty = (cy + dy).coerceIn(0f, 1f)
                cx = cx * smooth + tx * (1 - smooth)
                cy = cy * smooth + ty * (1 - smooth)
                outX = cx; outY = cy
            } else haveLast = false
        } else haveLast = false

        if (newState == state) { pending = null; pendingCount = 0 }
        else {
            if (pending == newState) pendingCount++ else { pending = newState; pendingCount = 1 }
            if (pendingCount >= minStable) {
                if (state != "HOLD" && newState == "HOLD") events += "HOLD_START"
                if (state == "HOLD" && newState != "HOLD") events += "HOLD_END"
                state = newState; pending = null; pendingCount = 0
            }
        }
        return EngineOut(state, PointF(outX, outY), events)
    }

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
        val v1x = a.x() - b.x(); val v1y = a.y() - b.y(); val v1z = a.z() - b.z()
        val v2x = c.x() - b.x(); val v2y = c.y() - b.y(); val v2z = c.z() - b.z()
        val num = (v1x * v2x + v1y * v2y + v1z * v2z).toDouble()
        val d1 = sqrt((v1x * v1x + v1y * v1y + v1z * v1z).toDouble())
        val d2 = sqrt((v2x * v2x + v2y * v2y + v2z * v2z).toDouble())
        val cos = (num / (d1 * d2)).coerceIn(-1.0, 1.0)
        return Math.toDegrees(acos(cos))
    }

    private fun dist(a: NormalizedLandmark, b: NormalizedLandmark): Float {
        val dx = a.x() - b.x(); val dy = a.y() - b.y(); val dz = a.z() - b.z()
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
}
