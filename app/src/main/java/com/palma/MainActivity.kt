package com.palma

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
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
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {

    private val TAG = "MP"

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
            .setMinHandDetectionConfidence(0.3f)  // lower threshold, more forgiving
            .setMinTrackingConfidence(0.3f)
            .setMinHandPresenceConfidence(0.3f)
            .setResultListener { result, _ ->
                try {
                    val out = engine.process(result)
                    if (out.cursor.x.isNaN() || out.cursor.y.isNaN()) return@setResultListener
                    runOnUiThread { renderAndAct(out) }
                } catch (t: Throwable) {
                    Log.e(TAG, "Result listener exception", t)
                }
            }
            .setErrorListener { e -> Log.e(TAG, "MediaPipe error: $e") }
            .build()

        landmarker = HandLandmarker.createFromOptions(this, opts)
        Log.i(TAG, "HandLandmarker initialized (LIVE_STREAM, adjusted thresholds)")
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().apply {
                setSurfaceProvider(previewView.surfaceProvider)
            }

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .setTargetResolution(Size(640, 480)) // optional but helps performance
                .build()
                .apply {
                    setAnalyzer(ContextCompat.getMainExecutor(this@MainActivity)) { proxy ->
                        try {
                            val mpImage = proxy.toMpImage(isFrontCamera)
                            landmarker.detectAsync(mpImage, SystemClock.uptimeMillis())
                        } catch (e: Exception) {
                            Log.e(TAG, "Frame analyze error", e)
                        } finally {
                            proxy.close()
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
    private var smoothFactor = 0.6f  // adjust 0.5–0.8 depending on stability

    private fun renderAndAct(out: EngineOut) {
        if (!::overlay.isInitialized || overlay.width == 0) return

        // No more (1 - x) for front camera
        val normalizedX = out.cursor.x.coerceIn(0f, 1f)
        val normalizedY = out.cursor.y.coerceIn(0f, 1f)

        val px = normalizedX * overlay.width
        val py = normalizedY * overlay.height
        overlay.updateCursor(px, py)

        val loc = IntArray(2)
        overlay.getLocationOnScreen(loc)
        val screenX = px + loc[0]
        val screenY = py + loc[1]

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

    // Converts YUV → ARGB with rotation + mirror correction
    private fun ImageProxy.toMpImage(isFrontCamera: Boolean): MPImage {
        val bmp = toBitmapWithRotationAndMirror(isFrontCamera)
        return BitmapImageBuilder(bmp).build()
    }

    // Core bitmap conversion from YUV to JPEG then to Bitmap
    private fun ImageProxy.toBitmapWithRotationAndMirror(isFrontCamera: Boolean): Bitmap {
        val nv21 = yuv420ToNv21(this)
        val yuv = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuv.compressToJpeg(Rect(0, 0, width, height), 90, out)
        val bytes = out.toByteArray()
        var bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

        // Rotate image according to camera orientation
        val matrix = Matrix()
        matrix.postRotate(imageInfo.rotationDegrees.toFloat())

        // Mirror horizontally if it's a front camera
        if (isFrontCamera) {
            matrix.postScale(-1f, 1f)
        }

        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        return bitmap
    }

}
