// Updated MainActivity.kt with automatic bar path tracking

package com.example.atry

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
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
import androidx.compose.foundation.border
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
import androidx.compose.ui.geometry.Size
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
import com.example.atry.detector.YOLOv11ObjectDetector
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

    Log.d("AutomaticCameraPreview", "Initializing automatic camera preview")

    val detector = remember {
        try {
            Log.d("AutomaticCameraPreview", "Creating YOLOv11 detector")
            YOLOv11ObjectDetector(
                context = context,
                modelPath = "optimizefloat16.tflite",
                confThreshold = 0.25f, // Slightly higher for more stable detection
                iouThreshold = 0.45f
            )
        } catch (e: Exception) {
            Log.e("AutomaticCameraPreview", "Failed to create detector: ${e.message}", e)
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
                    text = "Check if new model.tflite is in assets folder",
                    color = Color.Gray,
                    fontSize = 14.sp,
                    textAlign = TextAlign.Center
                )
            }
        }
        return
    }

    val previewView = remember { PreviewView(context) }

    // State variables for automatic tracking
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

    // Enhanced constants for automatic tracking
    val maxPathPoints = 300 // Reduced for better performance
    val minMovementThreshold = 0.015f // More sensitive
    val inactivityTimeoutMs = 3000L // 3 seconds of no detection = end session
    val minSessionDurationMs = 2000L // Minimum 2 seconds for a valid session
    val pathSegmentTimeoutMs = 1500L // 1.5 seconds = new path segment

    DisposableEffect(detector) {
        onDispose {
            try {
                detector.close()
                Log.d("AutomaticCameraPreview", "Detector closed successfully")
            } catch (e: Exception) {
                Log.e("AutomaticCameraPreview", "Error closing detector: ${e.message}", e)
            }
        }
    }

    // Camera setup
    LaunchedEffect(previewView) {
        try {
            Log.d("AutomaticCameraPreview", "Setting up camera")
            val cameraProvider = ProcessCameraProvider.getInstance(context).get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val imageAnalysis = androidx.camera.core.ImageAnalysis.Builder()
                .setBackpressureStrategy(androidx.camera.core.ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analyzer ->
                    analyzer.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                        if (!isProcessing) {
                            isProcessing = true

                            try {
                                val bitmap = BitmapUtils.imageProxyToBitmap(imageProxy)
                                val newDetections = detector.detect(bitmap)
                                detections = newDetections

                                val currentTime = System.currentTimeMillis()

                                // AUTOMATIC TRACKING LOGIC
                                if (newDetections.isNotEmpty()) {
                                    lastDetectionTime = currentTime

                                    // Start new session if not active
                                    if (!activeTrackingSession) {
                                        activeTrackingSession = true
                                        pathStartTime = currentTime
                                        sessionStartTime = currentTime
                                        barPaths = listOf(BarPath())
                                        Log.d("AutoTracking", "Started new automatic tracking session")
                                    }

                                    // Process bar path automatically
                                    val updatedData = processAutomaticBarPath(
                                        detections = newDetections,
                                        currentPaths = barPaths,
                                        currentTime = currentTime,
                                        pathStartTime = pathStartTime,
                                        maxPoints = maxPathPoints,
                                        minThreshold = minMovementThreshold,
                                        segmentTimeoutMs = pathSegmentTimeoutMs
                                    )

                                    barPaths = updatedData.paths
                                    currentMovement = updatedData.movement
                                    repCount = updatedData.repCount
                                    totalDistance = updatedData.totalDistance

                                    currentMovement?.let { movement ->
                                        allMovements = allMovements + movement
                                    }

                                } else {
                                    // Check for session timeout
                                    if (activeTrackingSession &&
                                        currentTime - lastDetectionTime > inactivityTimeoutMs) {

                                        val sessionDuration = currentTime - sessionStartTime
                                        if (sessionDuration > minSessionDurationMs) {
                                            Log.d("AutoTracking", "Session ended after ${sessionDuration}ms, reps: $repCount")
                                        }

                                        // End session but keep paths visible for a while
                                        activeTrackingSession = false
                                    }
                                }

                                // Calculate FPS
                                frameCount++
                                if (currentTime - lastFpsUpdate >= 1000) {
                                    fps = frameCount * 1000f / (currentTime - lastFpsUpdate)
                                    frameCount = 0
                                    lastFpsUpdate = currentTime
                                }

                            } catch (e: Exception) {
                                Log.e("AutomaticCameraPreview", "Error processing frame: ${e.message}", e)
                                cameraError = "Frame processing error: ${e.message}"
                            } finally {
                                isProcessing = false
                                imageProxy.close()
                            }
                        } else {
                            imageProxy.close()
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

            Log.d("AutomaticCameraPreview", "Camera bound successfully")
            cameraError = null

        } catch (e: Exception) {
            val errorMsg = "Camera setup failed: ${e.message}"
            Log.e("AutomaticCameraPreview", errorMsg, e)
            cameraError = errorMsg
            Toast.makeText(context, errorMsg, Toast.LENGTH_LONG).show()
        }
    }

    // Auto-cleanup paths every 30 seconds
    LaunchedEffect(barPaths) {
        delay(30000) // 30 seconds
        if (!activeTrackingSession && barPaths.isNotEmpty()) {
            val currentTime = System.currentTimeMillis()
            val cleanedPaths = barPaths.map { path ->
                if (path.points.isNotEmpty()) {
                    val cutoffTime = currentTime - 15000L // Keep last 15 seconds
                    val filteredPoints = path.points.filter { it.timestamp > cutoffTime }
                    path.copy(points = filteredPoints.toMutableList())
                } else {
                    path
                }
            }.filter { it.points.isNotEmpty() }

            if (cleanedPaths != barPaths) {
                barPaths = cleanedPaths
                Log.d("AutoTracking", "Auto-cleaned old path segments")
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
                    Log.d("AutoTracking", "Manually cleared all paths")
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
                                    Log.e("AutoTracking", "Excel report error", error)
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
                                    Log.e("AutoTracking", "CSV report error", error)
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

// Enhanced data class for automatic tracking results
data class AutomaticBarPathResult(
    val paths: List<BarPath>,
    val movement: MovementAnalysis?,
    val repCount: Int,
    val totalDistance: Float
)

// Enhanced automatic bar path processing with intelligent segmentation
private fun processAutomaticBarPath(
    detections: List<Detection>,
    currentPaths: List<BarPath>,
    currentTime: Long,
    pathStartTime: Long,
    maxPoints: Int,
    minThreshold: Float,
    segmentTimeoutMs: Long
): AutomaticBarPathResult {

    if (detections.isEmpty()) {
        return AutomaticBarPathResult(currentPaths, null, 0, 0f)
    }

    val detection = detections.first()
    val centerX = (detection.bbox.left + detection.bbox.right) / 2f
    val centerY = (detection.bbox.top + detection.bbox.bottom) / 2f
    val newPoint = PathPoint(centerX, centerY, currentTime)

    var workingPaths = currentPaths.toMutableList()
    var activePath = workingPaths.lastOrNull()

    // Create new path if none exists or if there's a time gap
    if (activePath == null ||
        (activePath.points.isNotEmpty() &&
                currentTime - activePath.points.last().timestamp > segmentTimeoutMs)) {

        activePath = BarPath(
            color = getPathColor(workingPaths.size),
            startTime = currentTime
        )
        workingPaths.add(activePath)
        Log.d("AutoPath", "Created new path segment #${workingPaths.size}")
    }

    // Check if movement is significant enough
    val shouldAddPoint = if (activePath.points.isEmpty()) {
        true
    } else {
        val lastPoint = activePath.points.last()
        val distance = newPoint.distanceTo(lastPoint)
        distance > minThreshold
    }

    if (!shouldAddPoint) {
        return AutomaticBarPathResult(workingPaths, null, 0, 0f)
    }

    // Add point with intelligent path management
    activePath.addPoint(newPoint, maxPoints)

    // Automatically trim old points if path gets too long
    if (activePath.points.size > maxPoints * 0.8) {
        val pointsToKeep = (maxPoints * 0.6).toInt()
        val trimmedPoints = activePath.points.takeLast(pointsToKeep)
        activePath.points.clear()
        activePath.points.addAll(trimmedPoints)
        Log.d("AutoPath", "Trimmed path to $pointsToKeep points")
    }

    // Analyze movement
    val movement = analyzeMovement(activePath.points)

    // Count reps across all paths
    val totalReps = workingPaths.map { path ->
        if (path.points.size > 20) countReps(path.points) else 0
    }.sum()

    // Calculate total distance
    val totalDistance = workingPaths.map { it.getTotalDistance() }.sum()

    Log.d("AutoPath", "Auto-processed: ${activePath.points.size} points, $totalReps reps, distance: $totalDistance")

    return AutomaticBarPathResult(workingPaths, movement, totalReps, totalDistance)
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
                    strokeWidth = 2.dp.toPx(),
                    pathEffect = if (pathIndex == paths.size - 1) {
                        // Solid line for current path
                        null
                    } else {
                        // Dashed line for older paths
                        PathEffect.dashPathEffect(floatArrayOf(10f, 5f), 0f)
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
                        radius = 2.dp.toPx(),
                        center = Offset(pointX, pointY)
                    )
                    drawCircle(
                        color = path.color,
                        radius = 2.dp.toPx(),
                        center = Offset(pointX, pointY)
                    )
                } else if (index % 10 == 0) {
                    // Key points every 10th point
                    drawCircle(
                        color = path.color.copy(alpha = alpha),
                        radius = 2.dp.toPx(),
                        center = Offset(pointX, pointY)
                    )
                }
            }
        }
    }
}

// Keep the existing drawDetections function unchanged
private fun DrawScope.drawDetections(
    detections: List<Detection>,
    detector: YOLOv11ObjectDetector
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
            radius = 2.dp.toPx(),
            center = Offset(centerX, centerY)
        )

        if (right > left + 10f && bottom > top + 10f) {
            drawRect(
                color = Color.Green,
                topLeft = Offset(left, top),
                size = Size(right - left, bottom - top),
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
                radius = 2.dp.toPx(),
                center = Offset(centerX, centerY),
                style = Stroke(width = 6.dp.toPx())
            )
        }
    }
}

// Keep existing analyzeMovement and countReps functions
private fun analyzeMovement(points: List<PathPoint>): MovementAnalysis? {
    if (points.size < 5) return null

    val recentPoints = points.takeLast(10)
    val totalVerticalMovement = recentPoints.zipWithNext { a, b -> b.y - a.y }.sum()
    val totalTime = recentPoints.last().timestamp - recentPoints.first().timestamp

    val velocity = if (totalTime > 0) abs(totalVerticalMovement) / (totalTime / 1000f) else 0f

    val direction = when {
        totalVerticalMovement > 0.02f -> MovementDirection.DOWN
        totalVerticalMovement < -0.02f -> MovementDirection.UP
        else -> MovementDirection.STABLE
    }

    val totalDistance = points.zipWithNext { a, b -> a.distanceTo(b) }.sum()

    return MovementAnalysis(
        direction = direction,
        velocity = velocity,
        acceleration = 0f,
        totalDistance = totalDistance,
        repCount = 0
    )
}

private fun countReps(points: List<PathPoint>): Int {
    if (points.size < 20) return 0

    var repCount = 0
    var lastDirection: MovementDirection? = null
    var inUpPhase = false
    val smoothingWindow = 5
    val repThreshold = 0.05f

    for (i in smoothingWindow until points.size - smoothingWindow) {
        val beforeY = points.subList(i - smoothingWindow, i).map { it.y }.average().toFloat()
        val afterY = points.subList(i, i + smoothingWindow).map { it.y }.average().toFloat()

        val currentDirection = when {
            afterY - beforeY > 0.02f -> MovementDirection.DOWN
            afterY - beforeY < -0.02f -> MovementDirection.UP
            else -> MovementDirection.STABLE
        }

        if (lastDirection == MovementDirection.UP && currentDirection == MovementDirection.DOWN && inUpPhase) {
            val upStartIndex = findLastDirectionChange(points, i, MovementDirection.DOWN, MovementDirection.UP, smoothingWindow)
            if (upStartIndex != -1) {
                val displacement = abs(points[i].y - points[upStartIndex].y)
                if (displacement > repThreshold) {
                    repCount++
                    inUpPhase = false
                    Log.d("AutoRepCounter", "Rep detected! Count: $repCount, Displacement: $displacement")
                }
            }
        }

        if (lastDirection == MovementDirection.DOWN && currentDirection == MovementDirection.UP) {
            inUpPhase = true
        }

        if (currentDirection != MovementDirection.STABLE) {
            lastDirection = currentDirection
        }
    }

    return repCount
}

private fun findLastDirectionChange(
    points: List<PathPoint>,
    currentIndex: Int,
    fromDirection: MovementDirection,
    toDirection: MovementDirection,
    smoothingWindow: Int
): Int {
    for (i in currentIndex - 1 downTo smoothingWindow) {
        val beforeY = points.subList(i - smoothingWindow, i).map { it.y }.average().toFloat()
        val afterY = points.subList(i, i + smoothingWindow).map { it.y }.average().toFloat()

        val direction = when {
            afterY - beforeY > 0.02f -> MovementDirection.DOWN
            afterY - beforeY < -0.02f -> MovementDirection.UP
            else -> MovementDirection.STABLE
        }

        if (direction == fromDirection) {
            return i
        }
    }
    return -1
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
            }

            Spacer(modifier = Modifier.height(4.dp))

            // Control buttons - only Clear and Report generation
            Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.padding(horizontal = 8.dp)
            ) {
                Button(
                    onClick = onClearPaths,
                    modifier = Modifier.height(32.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color.Red)
                ) {
                    Text("Clear", fontSize = 11.sp, color = Color.White)
                }

                // Report generation buttons - only show if we have data
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
                                text = if (isGeneratingReport) "..." else "Excel",
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
                                text = if (isGeneratingReport) "..." else "CSV",
                                fontSize = 11.sp,
                                color = Color.White
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(6.dp))

            // Status information with automatic tracking indicators


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


            // Instructions for automatic mode

        }
    }
}