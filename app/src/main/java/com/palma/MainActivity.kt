package com.palma

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker

import com.palma.gesture.EngineOut
import com.palma.gesture.GestureEngine
import com.palma.ui.CursorOverlay
import com.palma.access.PalmaService

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import java.io.ByteArrayOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: CursorOverlay
    private lateinit var landmarker: HandLandmarker
    private val engine = GestureEngine()

    private val requestCamera =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) safeInitAndStart()
            else Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlay = findViewById(R.id.cursorOverlay)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            safeInitAndStart()
        } else {
            requestCamera.launch(Manifest.permission.CAMERA)
        }
    }

    private fun safeInitAndStart() {
        try {
            setupLandmarker()
            startCamera()
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Failed to initialize camera or model", Toast.LENGTH_LONG).show()
        }
    }

    private fun setupLandmarker() {
        val base = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .build()

        val opts = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(1)
            .build()

        landmarker = HandLandmarker.createFromOptions(this, opts)
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()

                val preview = Preview.Builder().build().apply {
                    setSurfaceProvider(previewView.surfaceProvider)
                }

                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build().apply {
                        setAnalyzer(ContextCompat.getMainExecutor(this@MainActivity)) { proxy ->
                            try {
                                proxy.image?.let {
                                    val mpImg = proxy.toMpImage()
                                    val io = ImageProcessingOptions.builder()
                                        .setRotationDegrees(proxy.imageInfo.rotationDegrees)
                                        .build()
                                    val ts = SystemClock.uptimeMillis()
                                    val result = landmarker.detectForVideo(mpImg, io, ts)
                                    val out = engine.process(result)
                                    renderAndAct(out)
                                }
                            } catch (t: Throwable) {
                                t.printStackTrace()
                            } finally {
                                proxy.close()
                            }
                        }
                    }

                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)

            } catch (t: Throwable) {
                t.printStackTrace()
                Toast.makeText(this, "Camera start failed: ${t.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun renderAndAct(out: EngineOut) {
        // ensure overlay initialized
        if (!::overlay.isInitialized || overlay.width == 0 || overlay.height == 0) return

        // view-local coordinates (0..width, 0..height)
        val vx = out.cursor.x * overlay.width
        val vy = out.cursor.y * overlay.height
        overlay.updateCursor(vx, vy)

        // convert view-local to screen coordinates
        val loc = IntArray(2)
        overlay.getLocationOnScreen(loc) // fills loc[0]=x, loc[1]=y
        val screenX = loc[0] + vx
        val screenY = loc[1] + vy

        // debug log (remove/disable after confirming)
        android.util.Log.d("MainActivity", "cursor view=($vx,$vy) screen=($screenX,$screenY) state=${out.state} events=${out.events}")

        PalmaService.instance?.let { svc ->
            if ("HOLD_START" in out.events) svc.downAt(screenX, screenY)
            if (out.state == "HOLD")        svc.moveTo(screenX, screenY)
            if ("HOLD_END" in out.events)   svc.up()
        }
    }

    // --- ImageProxy → MPImage helpers ---
    private fun ImageProxy.toMpImage(): MPImage {
        val bmp = toBitmapCompat()
        return BitmapImageBuilder(bmp).build()
    }

    private fun ImageProxy.toBitmapCompat(): Bitmap {
        return try {
            val nv21 = yuv420ToNv21(this)
            val yuv = YuvImage(nv21, ImageFormat.NV21, width, height, null)
            val out = ByteArrayOutputStream()
            yuv.compressToJpeg(Rect(0, 0, width, height), 80, out)
            val bytes = out.toByteArray()
            android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        } catch (t: Throwable) {
            Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888)
        }
    }

    private fun yuv420ToNv21(image: ImageProxy): ByteArray {
        val y = image.planes[0].buffer
        val u = image.planes[1].buffer
        val v = image.planes[2].buffer
        val ySize = y.remaining()
        val uSize = u.remaining()
        val vSize = v.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        y.get(nv21, 0, ySize)
        v.get(nv21, ySize, vSize)
        u.get(nv21, ySize + vSize, uSize)
        return nv21
    }
}
