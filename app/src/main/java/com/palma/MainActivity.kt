package com.palma

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.SystemClock
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.app.ActivityCompat

// MediaPipe
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker

// Your classes
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
            if (granted) {
                safeInitAndStart()
            }
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
        } catch (t: Throwable) {
            t.printStackTrace()
            return
        }
        startCamera()
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
            val provider = future.get()

            val preview = Preview.Builder().build().apply {
                setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build().apply {
                    setAnalyzer(ContextCompat.getMainExecutor(this@MainActivity)) { proxy ->
                        try {
                            if (proxy.image == null) { proxy.close(); return@setAnalyzer }
                            val mpImg = proxy.toMpImage()
                            val io = ImageProcessingOptions.builder()
                                .setRotationDegrees(proxy.imageInfo.rotationDegrees)
                                .build()
                            val ts = SystemClock.uptimeMillis()
                            val result = landmarker.detectForVideo(mpImg, io, ts)
                            val out = engine.process(result)
                            renderAndAct(out)
                        } catch (t: Throwable) {
                            t.printStackTrace()
                        } finally {
                            proxy.close()
                        }
                    }
                }

            provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun renderAndAct(out: EngineOut) {
        val px = out.cursor.x * overlay.width
        val py = out.cursor.y * overlay.height
        overlay.updateCursor(px, py)

        PalmaService.instance?.let { svc ->
            if ("HOLD_START" in out.events) svc.downAt(px, py)
            if (out.state == "HOLD")        svc.moveTo(px, py)
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
        val ySize = y.remaining(); val uSize = u.remaining(); val vSize = v.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        y.get(nv21, 0, ySize)
        v.get(nv21, ySize, vSize)
        u.get(nv21, ySize + vSize, uSize)
        return nv21
    }
}
