package com.example.atry

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.util.Size // Import Android's Size class
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.atry.detector.BitmapUtils
import com.example.atry.detector.Detection
import com.example.atry.detector.NewYOLODetector
import com.example.atry.detector.BarPathAnalyzer
import com.example.atry.detector.ReportGenerator
import com.example.atry.detector.PathPoint
import com.example.atry.detector.BarPath
import com.example.atry.detector.MovementDirection
import com.example.atry.detector.MovementAnalysis
import com.example.atry.ui.theme.TryTheme
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.sqrt

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d(TAG, "MainActivity onCreate started")

        setContent {
            TryTheme {
                Surface(color = MaterialTheme.colorScheme.background) {
                    MainContent()
                }
            }
        }
    }

    @Composable
    private fun MainContent() {
        val context = LocalContext.current

        var hasCameraPermission by remember {
            mutableStateOf(
                ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED
            )
        }

        var permissionRequested by remember { mutableStateOf(false) }
        var permissionDenied by remember { mutableStateOf(false) }

        val permissionLauncher = rememberLauncherForActivityResult(
            contract = ActivityResultContracts.RequestPermission()
        ) { isGranted ->
            Log.d(TAG, "Camera permission result: $isGranted")
            hasCameraPermission = isGranted
            permissionRequested = true

            if (!isGranted) {
                permissionDenied = true
                Toast.makeText(context, "Camera permission is required for this app", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(context, "Camera permission granted", Toast.LENGTH_SHORT).show()
            }
        }

        LaunchedEffect(Unit) {
            if (!hasCameraPermission && !permissionRequested) {
                Log.d(TAG, "Requesting camera permission")
                delay(300)
                permissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }

        Box(modifier = Modifier.fillMaxSize()) {
            when {
                hasCameraPermission -> {
                    Log.d(TAG, "Camera permission granted, showing camera preview")
                    AutomaticCameraPreviewWithYOLOv11()
                }
                permissionDenied -> {
                    PermissionDeniedScreen {
                        permissionDenied = false
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                }
                else -> {
                    PermissionRequestScreen {
                        permissionLauncher.launch(Manifest.permission.CAMERA)
                    }
                }
            }
        }
    }
}

@Composable
private fun PermissionRequestScreen(onRequest: () -> Unit) {
    val pulse by animateDpAsState(
        targetValue = 120.dp,
        animationSpec = tween(800, easing = FastOutSlowInEasing),
        label = "pulse_animation"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF1E1E1E))
            .pointerInput(Unit) {
                detectTapGestures { onRequest() }
            },
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "📱",
                fontSize = 48.sp,
                color = Color.White
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = "Camera Permission Required",
                style = MaterialTheme.typography.titleLarge,
                color = Color.White,
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "This app needs camera access to detect barbells",
                style = MaterialTheme.typography.bodyMedium,
                color = Color.Gray,
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(24.dp))

            Button(
                onClick = onRequest,
                modifier = Modifier
                    .size(pulse)
                    .clip(RoundedCornerShape(16.dp))
            ) {
                Text(
                    text = "Grant Permission",
                    color = Color.White,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
private fun PermissionDeniedScreen(onRetry: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFF1E1E1E)),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "❌",
                fontSize = 48.sp,
                color = Color.Red
            )
            Spacer(modifier = Modifier.height(16.dp))
            Text(
                text = "Camera Permission Denied",
                style = MaterialTheme.typography.titleLarge,
                color = Color.White,
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Please grant camera permission in Settings or tap below to try again",
                style = MaterialTheme.typography.bodyMedium,
                color = Color.Gray,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(horizontal = 32.dp)
            )
            Spacer(modifier = Modifier.height(24.dp))

            Button(
                onClick = onRetry,
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Try Again",
                    color = Color.White,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
private fun AutomaticCameraPreviewWithYOLOv11() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    Log.d("HighFPS", "Initializing high-FPS camera preview")

    val detector = remember {
        try {
            Log.d("HighFPS", "🚀 Creating BRAND NEW detector to replace old one...")

            // Check if model file exists
            val modelExists = try {
                context.assets.open("optimizefloat16.tflite").use { true }
            } catch (e: Exception) {
                Log.e("HighFPS", "Model file 'optimizefloat16.tflite' not found in assets: ${e.message}")
                false
            }

            if (!modelExists) {
                Log.e("HighFPS", "❌ Model file missing! Please ensure 'optimizefloat16.tflite' is in src/main/assets/")
                Toast.makeText(context, "Model file 'optimizefloat16.tflite' not found in assets folder", Toast.LENGTH_LONG).show()
                null
            } else {
                Log.d("HighFPS", "✅ Model file found, creating BRAND NEW detector...")

                // EXPLICITLY create NewYOLODetector to replace any old ones
                val newDetector = NewYOLODetector(
                    context = context,
                    modelPath = "optimizefloat16.tflite",
                    confThreshold = 0.05f, // Very low threshold for testing
                    iouThreshold = 0.45f
                )

                Log.d("HighFPS", "✅ BRAND NEW detector created successfully!")
                newDetector
            }
        } catch (e: Exception) {
            Log.e("HighFPS", "Failed to create detector: ${e.message}", e)
            Toast.makeText(context, "Failed to load AI model: ${e.message}", Toast.LENGTH_LONG).show()
            null
        }
    }

    val analyzer = remember { BarPathAnalyzer() }
    val reportGenerator = remember { ReportGenerator(context) }

    if (detector == null) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = "⚠️ Model Loading Failed",
                    color = Color.Red,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Check if optimizefloat16.tflite is in assets folder",
                    color = Color.Gray,
                    fontSize = 14.sp,
                    textAlign = TextAlign.Center
                )
            }
        }
        return
    }

    val previewView = remember { PreviewView(context) }

    // State variables for high-FPS automatic tracking
    var detections by remember { mutableStateOf<List<Detection>>(emptyList()) }
    var isProcessing by remember { mutableStateOf(false) }
    var fps by remember { mutableStateOf(0f) }
    var cameraError by remember { mutableStateOf<String?>(null) }

    // Enhanced automatic bar path tracking state
    var barPaths by remember { mutableStateOf<List<BarPath>>(listOf()) }
    var currentMovement by remember { mutableStateOf<MovementAnalysis?>(null) }
    var repCount by remember { mutableStateOf(0) }
    var totalDistance by remember { mutableStateOf(0f) }
    var sessionStartTime by remember { mutableStateOf(System.currentTimeMillis()) }
    var allMovements by remember { mutableStateOf<List<MovementAnalysis>>(emptyList()) }
    var isGeneratingReport by remember { mutableStateOf(false) }

    // Automatic tracking state
    var lastDetectionTime by remember { mutableStateOf(0L) }
    var activeTrackingSession by remember { mutableStateOf(false) }
    var pathStartTime by remember { mutableStateOf(0L) }

    // FPS calculation
    var frameCount by remember { mutableStateOf(0) }
    var lastFpsUpdate by remember { mutableStateOf(System.currentTimeMillis()) }

    // BALANCED optimization constants
    val maxPathPoints = 200 // Increased for better tracking
    val minMovementThreshold = 0.015f // More sensitive
    val inactivityTimeoutMs = 3500L // Longer timeout
    val minSessionDurationMs = 2000L // Longer minimum duration

    // FPS optimization variables
    var cachedDetections by remember { mutableStateOf<List<Detection>>(emptyList()) }

    DisposableEffect(detector) {
        onDispose {
            try {
                detector.close()
                Log.d("HighFPS", "Detector closed successfully")
            } catch (e: Exception) {
                Log.e("HighFPS", "Error closing detector: ${e.message}", e)
            }
        }
    }

    // HIGH-FPS Camera setup with aggressive optimizations
    LaunchedEffect(previewView) {
        try {
            Log.d("HighFPS", "Setting up high-FPS camera")
            val cameraProvider = ProcessCameraProvider.getInstance(context).get()

            val preview = Preview.Builder()
                .setTargetResolution(Size(640, 640))  // Better resolution for detection
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val imageAnalysis = androidx.camera.core.ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 640))  // Better resolution for detection
                .setBackpressureStrategy(androidx.camera.core.ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setImageQueueDepth(2) // Slightly larger queue
                .build()
                .also { analyzer ->
                    analyzer.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->

                        frameCount++
                        val currentTime = System.currentTimeMillis()

                        // BALANCED FRAME PROCESSING - less aggressive for better detection
                        val shouldProcess = when {
                            fps < 10f -> frameCount % 3 == 0   // Moderate skipping when struggling
                            fps < 15f -> frameCount % 2 == 0   // Light skipping
                            else -> true                       // Process all frames when FPS is good
                        }

                        if (shouldProcess && !isProcessing) {
                            isProcessing = true

                            try {
                                // PROPER PROCESSING with correct input size - USING NEW DETECTOR
                                val bitmap = BitmapUtils.imageProxyToBitmap(imageProxy)
                                val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true) // Must match model input!

                                Log.d("NEW_DETECTOR", "🎯 Using NEW detector for inference...")
                                val newDetections = detector.detect(scaledBitmap)
                                Log.d("NEW_DETECTOR", "✅ NEW detector returned ${newDetections.size} detections")

                                // Debug logging every 30 frames
                                if (frameCount % 30 == 0) {
                                    Log.d("Detection", "===== FRAME $frameCount DEBUG =====")
                                    Log.d("Detection", "Input bitmap: ${bitmap.width}x${bitmap.height}")
                                    Log.d("Detection", "Scaled bitmap: ${scaledBitmap.width}x${scaledBitmap.height}")
                                    Log.d("Detection", "Found ${newDetections.size} detections")
                                    newDetections.forEachIndexed { idx, detection ->
                                        Log.d("Detection", "Detection $idx: Confidence=${String.format("%.3f", detection.score)}")
                                        Log.d("Detection", "  BBox: left=${String.format("%.3f", detection.bbox.left)}, top=${String.format("%.3f", detection.bbox.top)}, right=${String.format("%.3f", detection.bbox.right)}, bottom=${String.format("%.3f", detection.bbox.bottom)}")
                                        Log.d("Detection", "  Size: width=${String.format("%.3f", detection.bbox.right - detection.bbox.left)}, height=${String.format("%.3f", detection.bbox.bottom - detection.bbox.top)}")
                                    }
                                    Log.d("Detection", "================================")
                                }

                                // Update detections every time we process
                                cachedDetections = newDetections
                                detections = newDetections

                                // FAST TRACKING LOGIC
                                if (newDetections.isNotEmpty()) {
                                    lastDetectionTime = currentTime

                                    if (!activeTrackingSession) {
                                        activeTrackingSession = true
                                        pathStartTime = currentTime
                                        sessionStartTime = currentTime
                                        barPaths = listOf(BarPath())
                                        Log.d("HighFPS", "Fast tracking session started")
                                    }

                                    // ULTRA-FAST PATH PROCESSING
                                    val updatedData = processFastBarPath(
                                        detections = newDetections,
                                        currentPaths = barPaths,
                                        currentTime = currentTime,
                                        maxPoints = maxPathPoints,
                                        minThreshold = minMovementThreshold
                                    )

                                    barPaths = updatedData.paths
                                    currentMovement = updatedData.movement
                                    repCount = updatedData.repCount
                                    totalDistance = updatedData.totalDistance

                                    currentMovement?.let { movement ->
                                        allMovements = allMovements + movement
                                    }

                                } else {
                                    // Quick timeout check
                                    if (activeTrackingSession &&
                                        currentTime - lastDetectionTime > inactivityTimeoutMs) {
                                        activeTrackingSession = false
                                        Log.d("HighFPS", "Fast session ended")
                                    }
                                }

                                // EFFICIENT FPS calculation
                                if (currentTime - lastFpsUpdate >= 1000) {
                                    fps = frameCount * 1000f / (currentTime - lastFpsUpdate)
                                    frameCount = 0
                                    lastFpsUpdate = currentTime
                                    Log.d("HighFPS", "FPS: ${String.format("%.1f", fps)}")
                                }

                            } catch (e: Exception) {
                                Log.e("HighFPS", "Fast processing error: ${e.message}")
                            } finally {
                                isProcessing = false
                                imageProxy.close()
                            }
                        } else {
                            // SKIP FRAME - use cached detections for smooth UI
                            detections = cachedDetections
                            imageProxy.close()

                            // Still update FPS counter for skipped frames
                            if (currentTime - lastFpsUpdate >= 1000) {
                                fps = frameCount * 1000f / (currentTime - lastFpsUpdate)
                                frameCount = 0
                                lastFpsUpdate = currentTime
                            }
                        }
                    }
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis
            )

            Log.d("HighFPS", "High-FPS camera setup completed successfully")
            cameraError = null

        } catch (e: Exception) {
            val errorMsg = "High-FPS camera setup failed: ${e.message}"
            Log.e("HighFPS", errorMsg, e)
            cameraError = errorMsg
            Toast.makeText(context, errorMsg, Toast.LENGTH_LONG).show()
        }
    }

    // Auto-cleanup paths every 30 seconds
    LaunchedEffect(barPaths) {
        delay(30000)
        if (!activeTrackingSession && barPaths.isNotEmpty()) {
            val currentTime = System.currentTimeMillis()
            val cleanedPaths = barPaths.map { path ->
                if (path.points.isNotEmpty()) {
                    val cutoffTime = currentTime - 15000L
                    val filteredPoints = path.points.filter { it.timestamp > cutoffTime }
                    path.copy(points = filteredPoints.toMutableList())
                } else {
                    path
                }
            }.filter { it.points.isNotEmpty() }

            if (cleanedPaths != barPaths) {
                barPaths = cleanedPaths
                Log.d("HighFPS", "Auto-cleaned old path segments")
            }
        }
    }

    // UI Layout
    Box(modifier = Modifier.fillMaxSize()) {
        if (cameraError != null) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "📷 Camera Error",
                        color = Color.Red,
                        fontSize = 18.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = cameraError!!,
                        color = Color.Gray,
                        fontSize = 12.sp,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(16.dp)
                    )
                    Button(onClick = { cameraError = null }) {
                        Text("Retry")
                    }
                }
            }
        } else {
            AndroidView(
                factory = { previewView },
                modifier = Modifier.fillMaxSize()
            )

            Canvas(modifier = Modifier.fillMaxSize()) {
                drawDetections(detections, detector)
                drawAutomaticBarPaths(barPaths)
            }

            AutomaticInfoPanel(
                detections = detections,
                fps = fps,
                isProcessing = isProcessing,
                currentMovement = currentMovement,
                repCount = repCount,
                totalDistance = totalDistance,
                activeTrackingSession = activeTrackingSession,
                isGeneratingReport = isGeneratingReport,
                onClearPaths = {
                    barPaths = listOf()
                    repCount = 0
                    totalDistance = 0f
                    currentMovement = null
                    allMovements = emptyList()
                    activeTrackingSession = false
                    Log.d("HighFPS", "Manually cleared all paths")
                },
                onGenerateExcelReport = {
                    scope.launch {
                        isGeneratingReport = true
                        try {
                            val session = ReportGenerator.WorkoutSession(
                                startTime = sessionStartTime,
                                endTime = System.currentTimeMillis(),
                                totalReps = repCount,
                                paths = barPaths,
                                movements = allMovements
                            )

                            val result = reportGenerator.generateExcelReport(session, analyzer)
                            result.fold(
                                onSuccess = { file ->
                                    Toast.makeText(context, "Excel report generated: ${file.name}", Toast.LENGTH_LONG).show()
                                    reportGenerator.shareReport(file)
                                },
                                onFailure = { error ->
                                    Toast.makeText(context, "Error generating Excel report: ${error.message}", Toast.LENGTH_LONG).show()
                                    Log.e("HighFPS", "Excel report error", error)
                                }
                            )
                        } finally {
                            isGeneratingReport = false
                        }
                    }
                },
                onGenerateCSVReport = {
                    scope.launch {
                        isGeneratingReport = true
                        try {
                            val session = ReportGenerator.WorkoutSession(
                                startTime = sessionStartTime,
                                endTime = System.currentTimeMillis(),
                                totalReps = repCount,
                                paths = barPaths,
                                movements = allMovements
                            )

                            val result = reportGenerator.generateCSVReport(session, analyzer)
                            result.fold(
                                onSuccess = { file ->
                                    Toast.makeText(context, "CSV report generated: ${file.name}", Toast.LENGTH_LONG).show()
                                    reportGenerator.shareReport(file)
                                },
                                onFailure = { error ->
                                    Toast.makeText(context, "Error generating CSV report: ${error.message}", Toast.LENGTH_LONG).show()
                                    Log.e("HighFPS", "CSV report error", error)
                                }
                            )
                        } finally {
                            isGeneratingReport = false
                        }
                    }
                },
                modifier = Modifier.align(Alignment.TopStart)
            )
        }
    }
}

// Data classes
data class AutomaticBarPathResult(
    val paths: List<BarPath>,
    val movement: MovementAnalysis?,
    val repCount: Int,
    val totalDistance: Float
)

// ULTRA-FAST BAR PATH PROCESSING for high FPS
private fun processFastBarPath(
    detections: List<Detection>,
    currentPaths: List<BarPath>,
    currentTime: Long,
    maxPoints: Int,
    minThreshold: Float
): AutomaticBarPathResult {

    if (detections.isEmpty()) {
        return AutomaticBarPathResult(currentPaths, null, 0, 0f)
    }

    // SINGLE detection processing for maximum speed
    val detection = detections.first()
    val centerX = (detection.bbox.left + detection.bbox.right) / 2f
    val centerY = (detection.bbox.top + detection.bbox.bottom) / 2f
    val newPoint = PathPoint(centerX, centerY, currentTime)

    var workingPaths = currentPaths.toMutableList()
    var activePath = workingPaths.lastOrNull() ?: BarPath().also { workingPaths.add(it) }

    // FAST distance check using Manhattan distance (faster than Euclidean)
    val shouldAddPoint = if (activePath.points.isEmpty()) {
        true
    } else {
        val lastPoint = activePath.points.last()
        val distance = abs(newPoint.x - lastPoint.x) + abs(newPoint.y - lastPoint.y)
        distance > minThreshold
    }

    if (!shouldAddPoint) {
        return AutomaticBarPathResult(workingPaths, null, 0, 0f)
    }

    // ADD point with immediate aggressive trimming
    activePath.addPoint(newPoint, maxPoints)

    if (activePath.points.size > maxPoints) {
        val keepCount = maxPoints * 2 / 3
        val trimmedPoints = activePath.points.takeLast(keepCount)
        activePath.points.clear()
        activePath.points.addAll(trimmedPoints)
    }

    // FAST analysis using optimized functions
    val movement = fastAnalyzeMovement(activePath.points)
    val totalReps = fastRepCount(activePath.points)
    val totalDistance = activePath.getTotalDistance()

    return AutomaticBarPathResult(workingPaths, movement, totalReps, totalDistance)
}

// ULTRA-FAST movement analysis
private fun fastAnalyzeMovement(points: List<PathPoint>): MovementAnalysis? {
    if (points.size < 3) return null

    val recent = points.takeLast(4) // Minimal analysis window
    if (recent.size < 2) return null

    val verticalChange = recent.last().y - recent.first().y
    val timeSpan = (recent.last().timestamp - recent.first().timestamp) / 1000f

    val direction = when {
        verticalChange > 0.02f -> MovementDirection.DOWN
        verticalChange < -0.02f -> MovementDirection.UP
        else -> MovementDirection.STABLE
    }

    val velocity = if (timeSpan > 0) abs(verticalChange) / timeSpan else 0f
    val totalDistance = recent.zipWithNext { a, b -> a.distanceTo(b) }.sum()

    return MovementAnalysis(
        direction = direction,
        velocity = velocity,
        acceleration = 0f,
        totalDistance = totalDistance,
        repCount = 0
    )
}

// ULTRA-FAST rep counting with achievable thresholds
private fun fastRepCount(points: List<PathPoint>): Int {
    if (points.size < 8) return 0

    var repCount = 0
    var isInRep = false
    var startY: Float? = null
    var peakY: Float? = null

    val minMovement = 0.05f // REDUCED threshold - only 5% of screen
    val step = 2 // Process every 2nd point for maximum speed

    for (i in step until points.size step step) {
        val currentY = points[i].y

        // SIMPLE direction calculation using just 2 points
        val prevY = points[maxOf(0, i - step)].y
        val direction = when {
            currentY - prevY > 0.008f -> MovementDirection.DOWN
            currentY - prevY < -0.008f -> MovementDirection.UP
            else -> MovementDirection.STABLE
        }

        when {
            // START rep: detect upward movement
            !isInRep && direction == MovementDirection.UP -> {
                isInRep = true
                startY = currentY
                peakY = currentY
            }

            // TRACK peak position
            isInRep && currentY > (peakY ?: 0f) -> {
                peakY = currentY
            }

            // COMPLETE rep: downward movement with sufficient range
            isInRep && direction == MovementDirection.DOWN && peakY != null && startY != null -> {
                val totalMovement = peakY!! - startY!!
                val returnedToStart = currentY <= startY!! + (totalMovement * 0.4f) // Very forgiving

                if (totalMovement >= minMovement && returnedToStart) {
                    repCount++
                    isInRep = false
                    Log.d("FastRep", "⚡ FAST REP! Count: $repCount, Movement: ${String.format("%.3f", totalMovement)}")
                    startY = currentY
                    peakY = null
                }
            }
        }
    }

    return repCount
}

// Generate different colors for path segments
private fun getPathColor(pathIndex: Int): Color {
    val colors = listOf(
        Color.Cyan,
        Color.Yellow,
        Color.Green,
        Color.Magenta,
        Color.Red,
        Color.Blue
    )
    return colors[pathIndex % colors.size]
}

// Enhanced drawing for automatic paths with fade effects
private fun DrawScope.drawAutomaticBarPaths(paths: List<BarPath>) {
    val canvasWidth = size.width
    val canvasHeight = size.height
    val currentTime = System.currentTimeMillis()

    paths.forEachIndexed { pathIndex, path ->
        val points = path.points
        if (points.size > 1) {

            // Draw path with time-based fade effect
            for (i in 0 until points.size - 1) {
                val startPoint = points[i]
                val endPoint = points[i + 1]

                val startX = startPoint.x * canvasWidth
                val startY = startPoint.y * canvasHeight
                val endX = endPoint.x * canvasWidth
                val endY = endPoint.y * canvasHeight

                // Time-based alpha (newer points are more opaque)
                val timeSincePoint = currentTime - endPoint.timestamp
                val maxAge = 10000L // 10 seconds
                val alpha = (1f - (timeSincePoint.toFloat() / maxAge)).coerceIn(0.3f, 1f)

                // Position-based alpha (recent points in sequence)
                val positionAlpha = (i.toFloat() / points.size) * 0.7f + 0.3f
                val finalAlpha = alpha * positionAlpha

                drawLine(
                    color = path.color.copy(alpha = finalAlpha),
                    start = Offset(startX, startY),
                    end = Offset(endX, endY),
                    strokeWidth = 4.dp.toPx(),
                    pathEffect = if (pathIndex == paths.size - 1) {
                        null // Solid line for current path
                    } else {
                        PathEffect.dashPathEffect(floatArrayOf(10f, 5f), 0f) // Dashed for older paths
                    }
                )
            }

            // Draw key points with enhanced visibility
            points.forEachIndexed { index, point ->
                val pointX = point.x * canvasWidth
                val pointY = point.y * canvasHeight

                val timeSincePoint = currentTime - point.timestamp
                val alpha = (1f - (timeSincePoint.toFloat() / 15000L)).coerceIn(0.2f, 1f)

                if (index == points.size - 1) {
                    // Current point - larger and pulsing
                    drawCircle(
                        color = Color.White,
                        radius = 12.dp.toPx(),
                        center = Offset(pointX, pointY)
                    )
                    drawCircle(
                        color = path.color,
                        radius = 8.dp.toPx(),
                        center = Offset(pointX, pointY)
                    )
                } else if (index % 10 == 0) {
                    // Key points every 10th point
                    drawCircle(
                        color = path.color.copy(alpha = alpha),
                        radius = 6.dp.toPx(),
                        center = Offset(pointX, pointY)
                    )
                }
            }
        }
    }
}

// Optimized detection drawing
private fun DrawScope.drawDetections(
    detections: List<Detection>,
    detector: NewYOLODetector
) {
    val canvasWidth = size.width
    val canvasHeight = size.height

    detections.forEachIndexed { index, detection ->
        val bbox = detection.bbox

        val left = (bbox.left * canvasWidth).coerceIn(0f, canvasWidth)
        val top = (bbox.top * canvasHeight).coerceIn(0f, canvasHeight)
        val right = (bbox.right * canvasWidth).coerceIn(0f, canvasWidth)
        val bottom = (bbox.bottom * canvasHeight).coerceIn(0f, canvasHeight)

        val centerX = (left + right) / 2f
        val centerY = (top + bottom) / 2f

        // Center point
        drawCircle(
            color = Color.White,
            radius = 3.dp.toPx(),
            center = Offset(centerX, centerY)
        )

        if (right > left + 10f && bottom > top + 10f) {
            drawRect(
                color = Color.Green,
                topLeft = Offset(left, top),
                size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                style = Stroke(width = 2.dp.toPx())
            )

            val label = "${detector.getClassLabel(detection.classId)}: ${String.format("%.2f", detection.score)}"
            val textSize = 16.sp.toPx()
            val textPadding = 8.dp.toPx()

            val labelTop = if (top > textSize + textPadding * 2f) {
                top - textSize - textPadding * 2f
            } else {
                bottom + textPadding
            }

            drawContext.canvas.nativeCanvas.apply {
                val paint = android.graphics.Paint().apply {
                    color = Color.Green.toArgb()
                    this.textSize = 20.sp.toPx()
                    isAntiAlias = true
                    isFakeBoldText = true
                }
                drawText(
                    label,
                    left + textPadding,
                    labelTop + textSize + textPadding,
                    paint
                )
            }
        } else {
            drawCircle(
                color = Color.Green,
                radius = 40.dp.toPx(),
                center = Offset(centerX, centerY),
                style = Stroke(width = 6.dp.toPx())
            )
        }
    }
}

@Composable
private fun AutomaticInfoPanel(
    detections: List<Detection>,
    fps: Float,
    isProcessing: Boolean,
    currentMovement: MovementAnalysis?,
    repCount: Int,
    totalDistance: Float,
    activeTrackingSession: Boolean,
    isGeneratingReport: Boolean,
    onClearPaths: () -> Unit,
    onGenerateExcelReport: () -> Unit,
    onGenerateCSVReport: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .padding(top = 50.dp)
    ) {
        Column(
            modifier = Modifier.align(Alignment.TopCenter),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // App title with automatic indicator
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center
            ) {
                Text(
                    text = "⚡ High-FPS Bar Tracker",
                    color = Color.White,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center
                )
                if (activeTrackingSession) {
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "●",
                        color = Color.Red,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            Spacer(modifier = Modifier.height(4.dp))

            // Control buttons
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.padding(horizontal = 8.dp)
            ) {
                Button(
                    onClick = onClearPaths,
                    modifier = Modifier.height(32.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color.Red)
                ) {
                    Text("🗑️ Clear", fontSize = 11.sp, color = Color.White)
                }

                // Report generation buttons
                AnimatedVisibility(visible = repCount > 0) {
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Button(
                            onClick = onGenerateExcelReport,
                            enabled = !isGeneratingReport,
                            modifier = Modifier.height(32.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color(0xFF007ACC),
                                disabledContainerColor = Color.Gray
                            )
                        ) {
                            Text(
                                text = if (isGeneratingReport) "..." else "📊 Excel",
                                fontSize = 11.sp,
                                color = Color.White
                            )
                        }
                        Button(
                            onClick = onGenerateCSVReport,
                            enabled = !isGeneratingReport,
                            modifier = Modifier.height(32.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color(0xFF228B22),
                                disabledContainerColor = Color.Gray
                            )
                        ) {
                            Text(
                                text = if (isGeneratingReport) "..." else "📋 CSV",
                                fontSize = 11.sp,
                                color = Color.White
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(6.dp))

            // Status information
            Text(
                text = "Auto Mode: ${if (activeTrackingSession) "TRACKING" else "STANDBY"}",
                color = if (activeTrackingSession) Color.Green else Color.Yellow,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )

            Text(
                text = "FPS: ${String.format("%.1f", fps)} | Detections: ${detections.size}",
                color = Color.White,
                fontSize = 10.sp,
                textAlign = TextAlign.Center
            )

            // Main metrics
            Row(
                horizontalArrangement = Arrangement.spacedBy(16.dp),
                modifier = Modifier.padding(vertical = 4.dp)
            ) {
                Text(
                    text = "Reps: $repCount",
                    color = Color.Cyan,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Distance: ${String.format("%.2f", totalDistance)}",
                    color = Color.Yellow,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold
                )
            }

            if (isGeneratingReport) {
                Text(
                    text = "📄 Generating Report...",
                    color = Color.Yellow,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center
                )
            }

            // Movement analysis
            currentMovement?.let { movement ->
                Spacer(modifier = Modifier.height(4.dp))
                Row(
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    modifier = Modifier.padding(horizontal = 16.dp)
                ) {
                    Text(
                        text = "${movement.direction}",
                        color = when (movement.direction) {
                            MovementDirection.UP -> Color.Green
                            MovementDirection.DOWN -> Color.Red
                            MovementDirection.STABLE -> Color.Yellow
                        },
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "V: ${String.format("%.2f", movement.velocity)}",
                        color = Color.White,
                        fontSize = 10.sp
                    )
                }
            }

            // Detection details
            if (detections.isNotEmpty()) {
                Spacer(modifier = Modifier.height(4.dp))
                detections.take(1).forEachIndexed { index, detection ->
                    Text(
                        text = "Barbell Conf: ${String.format("%.2f", detection.score)}",
                        color = Color.Cyan,
                        fontSize = 10.sp,
                        textAlign = TextAlign.Center
                    )
                }
            }

            // Instructions for automatic mode
            if (!activeTrackingSession && repCount == 0) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Point camera at barbell to start automatic tracking",
                    color = Color.Gray,
                    fontSize = 10.sp,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.padding(horizontal = 24.dp)
                )
            }
        }
    }
}