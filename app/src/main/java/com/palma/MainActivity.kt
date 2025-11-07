package com.palma

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.core.ImageProxy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.palma.access.PalmaService
import com.palma.gesture.EngineOut
import com.palma.gesture.GestureEngine
import com.palma.ui.CursorOverlay
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {

    private val TAG = "MP"
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    private lateinit var previewView: PreviewView
    private lateinit var overlay: CursorOverlay
    private lateinit var landmarker: HandLandmarker
    private val engine = GestureEngine()
    private val isFrontCamera = true

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
            Log.e(TAG, "Initialization failed", e)
            Toast.makeText(this, "Failed to initialize: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun setupLandmarker() {
        val base = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .build()

        val opts = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumHands(1)
            .setMinHandDetectionConfidence(0.2f)
            .setMinTrackingConfidence(0.2f)
            .setMinHandPresenceConfidence(0.2f)
            .setResultListener { result, _image ->
                try {
                    val out = engine.process(result)
                    if (!out.cursor.x.isNaN() && !out.cursor.y.isNaN()) {
                        runOnUiThread { renderAndAct(out) }
                    } else {
                        Log.w(TAG, "Engine produced NaN coords")
                    }
                } catch (t: Throwable) {
                    Log.e(TAG, "Result listener exception", t)
                }
            }
            .setErrorListener { e -> Log.e(TAG, "MediaPipe error: $e") }
            .build()

        landmarker = HandLandmarker.createFromOptions(this, opts)
        Log.i(TAG, "HandLandmarker initialized (LIVE_STREAM)")
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            val preview = Preview.Builder()
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            val analysis = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build().apply {
                    setAnalyzer(analysisExecutor) { imageProxy ->
                        try {
                            // Convert ImageProxy to Bitmap properly
                            val bmp = imageProxy.toBitmapWithRotationAndMirror(isFrontCamera)
                            val mpImage = BitmapImageBuilder(bmp).build()
                            val timestamp = SystemClock.uptimeMillis()
                            landmarker.detectAsync(mpImage, timestamp)
                        } catch (e: Exception) {
                            Log.e(TAG, "Analyzer error", e)
                        } finally {
                            imageProxy.close()
                        }
                    }

                }

            provider.unbindAll()
            val selector = if (isFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA else CameraSelector.DEFAULT_BACK_CAMERA
            provider.bindToLifecycle(this, selector, preview, analysis)
            Log.i(TAG, "Camera started (front=$isFrontCamera)")
        }, ContextCompat.getMainExecutor(this))
    }

    private var lastX = 0f
    private var lastY = 0f
    private var smoothFactor = 0.3f  // more responsive cursor

    private fun renderAndAct(out: EngineOut) {
        if (!::overlay.isInitialized || overlay.width == 0 || overlay.height == 0) return

        // Correct mirroring for front camera here only
        val normalizedX = if (isFrontCamera) 1f - out.cursor.x.coerceIn(0f, 1f)
                          else out.cursor.x.coerceIn(0f, 1f)
        val normalizedY = out.cursor.y.coerceIn(0f, 1f)

        if (normalizedX.isNaN() || normalizedY.isNaN()) return

        val pxView = normalizedX.coerceIn(0f, 1f) * overlay.width
        val pyView = normalizedY.coerceIn(0f, 1f) * overlay.height
        overlay.updateCursor(pxView, pyView)

        val loc = IntArray(2)
        overlay.getLocationOnScreen(loc)
        val screenX = pxView + loc[0]
        val screenY = pyView + loc[1]

        PalmaService.instance?.let { svc ->
            if ("HOLD_START" in out.events) svc.downAt(screenX, screenY)
            if (out.state == "HOLD")        svc.moveTo(screenX, screenY)
            if ("HOLD_END" in out.events)   svc.up()
        }
    }

    private fun yuv420ToNv21(image: ImageProxy): ByteArray {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        return nv21
    }

    private fun ImageProxy.toMpImage(isFrontCamera: Boolean): MPImage {
        val bmp = toBitmapWithRotationAndMirror(isFrontCamera)
        return BitmapImageBuilder(bmp).build()
    }

    // Rotation only, no mirror
    private fun ImageProxy.toBitmapWithRotationAndMirror(isFrontCamera: Boolean): Bitmap {
        val nv21 = yuv420ToNv21(this)
        val yuv = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, width, height), 90, out)
        val bytes = out.toByteArray()
        var bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

        val matrix = Matrix()
        matrix.postRotate(imageInfo.rotationDegrees.toFloat())

        // ❌ Removed mirror step here

        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        return bitmap
    }
}
