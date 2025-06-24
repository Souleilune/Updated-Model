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

        // Track permission state more robustly
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

        // Enhanced permission launcher with better error handling
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

        // Request permission on first launch
        LaunchedEffect(Unit) {
            if (!hasCameraPermission && !permissionRequested) {
                Log.d(TAG, "Requesting camera permission")
                delay(300) // Small delay to ensure UI is ready
                permissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }

        Box(modifier = Modifier.fillMaxSize()) {
            when {
                hasCameraPermission -> {
                    Log.d(TAG, "Camera permission granted, showing camera preview")
                    CameraPreviewWithYOLOv11()
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
private fun CameraPreviewWithYOLOv11() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    Log.d("CameraPreview", "Initializing camera preview")

    // Create detector with error handling
    val detector = remember {
        try {
            Log.d("CameraPreview", "Creating YOLOv11 detector")
            YOLOv11ObjectDetector(
                context = context,
                modelPath = "new model.tflite",
                confThreshold = 0.15f,
                iouThreshold = 0.45f
            )
        } catch (e: Exception) {
            Log.e("CameraPreview", "Failed to create detector: ${e.message}", e)
            Toast.makeText(context, "Failed to load AI model: ${e.message}", Toast.LENGTH_LONG).show()
            null
        }
    }

    // Create analyzer and report generator
    val analyzer = remember { BarPathAnalyzer() }
    val reportGenerator = remember { ReportGenerator(context) }

    // Early return if detector creation failed
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

    // PreviewView instance
    val previewView = remember { PreviewView(context) }

    // State variables
    var detections by remember { mutableStateOf<List<Detection>>(emptyList()) }
    var isProcessing by remember { mutableStateOf(false) }
    var fps by remember { mutableStateOf(0f) }
    var cameraError by remember { mutableStateOf<String?>(null) }

    // Enhanced bar path tracking state with session management
    var barPaths by remember { mutableStateOf<List<BarPath>>(listOf(BarPath())) }
    var currentMovement by remember { mutableStateOf<MovementAnalysis?>(null) }
    var repCount by remember { mutableStateOf(0) }
    var isRecording by remember { mutableStateOf(false) }

    // Session management state
    var sessionStartTime by remember { mutableStateOf(0L) }
    var sessionEndTime by remember { mutableStateOf(0L) }
    var allMovements by remember { mutableStateOf<List<MovementAnalysis>>(emptyList()) }
    var isGeneratingReport by remember { mutableStateOf(false) }

    // FPS calculation
    var frameCount by remember { mutableStateOf(0) }
    var lastFpsUpdate by remember { mutableStateOf(System.currentTimeMillis()) }

    // Constants for path tracking
    val maxPathPoints = 500
    val minMovementThreshold = 0.02f
    val stableThreshold = 0.01f

    // Dispose detector when composable is removed
    DisposableEffect(detector) {
        onDispose {
            try {
                detector.close()
                Log.d("CameraPreview", "Detector closed successfully")
            } catch (e: Exception) {
                Log.e("CameraPreview", "Error closing detector: ${e.message}", e)
            }
        }
    }

    // Camera setup with enhanced error handling
    LaunchedEffect(previewView) {
        try {
            Log.d("CameraPreview", "Setting up camera")
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

                                // Update detections on main thread
                                detections = newDetections

                                Log.d("CameraPreview", "Frame processed - Detections: ${newDetections.size}")

                                // Process bar path tracking ONLY when recording
                                if (isRecording && newDetections.isNotEmpty()) {
                                    Log.d("CameraPreview", "Processing bar path - Recording is ON")
                                    val updatedData = processBarPath(
                                        detections = newDetections,
                                        currentPaths = barPaths,
                                        maxPoints = maxPathPoints,
                                        minThreshold = minMovementThreshold
                                    )

                                    barPaths = updatedData.paths
                                    currentMovement = updatedData.movement
                                    repCount = updatedData.repCount

                                    // Add movement to session data
                                    currentMovement?.let { movement ->
                                        allMovements = allMovements + movement
                                    }
                                } else if (!isRecording) {
                                    Log.d("CameraPreview", "Not recording - skipping bar path processing")
                                }

                                // Calculate FPS
                                frameCount++
                                val currentTime = System.currentTimeMillis()
                                if (currentTime - lastFpsUpdate >= 1000) {
                                    fps = frameCount * 1000f / (currentTime - lastFpsUpdate)
                                    frameCount = 0
                                    lastFpsUpdate = currentTime
                                }

                            } catch (e: Exception) {
                                Log.e("CameraPreview", "Error processing frame: ${e.message}", e)
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

            // Bind camera with error handling
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                imageAnalysis
            )

            Log.d("CameraPreview", "Camera bound successfully")
            cameraError = null // Clear any previous errors

        } catch (e: Exception) {
            val errorMsg = "Camera setup failed: ${e.message}"
            Log.e("CameraPreview", errorMsg, e)
            cameraError = errorMsg
            Toast.makeText(context, errorMsg, Toast.LENGTH_LONG).show()
        }
    }

    // UI Layout
    Box(modifier = Modifier.fillMaxSize()) {
        if (cameraError != null) {
            // Show error state
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
                    Button(
                        onClick = {
                            cameraError = null
                            // Trigger camera restart by updating the LaunchedEffect
                        }
                    ) {
                        Text("Retry")
                    }
                }
            }
        } else {
            // Camera Preview
            AndroidView(
                factory = { previewView },
                modifier = Modifier.fillMaxSize()
            )

            // Detection and Path Overlay with Debug Features
            Canvas(modifier = Modifier.fillMaxSize()) {
                // Draw detections
                drawDetections(detections, detector)

                // Draw bar paths
                drawBarPaths(barPaths)
            }

            // Enhanced Info Panel with Report Generation
            EnhancedInfoPanel(
                detections = detections,
                fps = fps,
                isProcessing = isProcessing,
                currentMovement = currentMovement,
                repCount = repCount,
                isRecording = isRecording,
                isGeneratingReport = isGeneratingReport,
                onStartStopRecording = {
                    isRecording = !isRecording
                    Log.d("CameraPreview", "Recording toggled - isRecording: $isRecording")
                    if (isRecording) {
                        // Start fresh when recording starts
                        barPaths = listOf(BarPath())
                        repCount = 0
                        allMovements = emptyList()
                        sessionStartTime = System.currentTimeMillis()
                        Log.d("CameraPreview", "Started recording - cleared paths")
                    } else {
                        sessionEndTime = System.currentTimeMillis()
                        Log.d("CameraPreview", "Stopped recording")
                    }
                },
                onClearPath = {
                    barPaths = listOf(BarPath())
                    repCount = 0
                    currentMovement = null
                    allMovements = emptyList()
                    Log.d("CameraPreview", "Cleared all paths")
                },
                onGenerateExcelReport = {
                    scope.launch {
                        isGeneratingReport = true
                        try {
                            val session = ReportGenerator.WorkoutSession(
                                startTime = sessionStartTime,
                                endTime = if (sessionEndTime > 0) sessionEndTime else System.currentTimeMillis(),
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
                                    Log.e("CameraPreview", "Excel report error", error)
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
                                endTime = if (sessionEndTime > 0) sessionEndTime else System.currentTimeMillis(),
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
                                    Log.e("CameraPreview", "CSV report error", error)
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

// Keep existing drawing functions (drawDetections, drawBarPaths, etc.)
private fun DrawScope.drawDetections(
    detections: List<Detection>,
    detector: YOLOv11ObjectDetector
) {
    val canvasWidth = size.width
    val canvasHeight = size.height

    Log.d("DrawDetections", "Canvas size: ${canvasWidth}x${canvasHeight}, detections: ${detections.size}")

    detections.forEachIndexed { index, detection ->
        val bbox = detection.bbox

        // Log the original normalized coordinates
        Log.d("DrawDetections", "Detection $index: Normalized bbox: left=${bbox.left}, top=${bbox.top}, right=${bbox.right}, bottom=${bbox.bottom}, conf=${detection.score}")

        // Convert normalized coordinates to pixel coordinates with bounds checking
        val left = (bbox.left * canvasWidth).coerceIn(0f, canvasWidth)
        val top = (bbox.top * canvasHeight).coerceIn(0f, canvasHeight)
        val right = (bbox.right * canvasWidth).coerceIn(0f, canvasWidth)
        val bottom = (bbox.bottom * canvasHeight).coerceIn(0f, canvasHeight)

        // Log the pixel coordinates
        Log.d("DrawDetections", "Detection $index: Pixel bbox: left=$left, top=$top, right=$right, bottom=$bottom")

        // Calculate center point
        val centerX = (left + right) / 2f
        val centerY = (top + bottom) / 2f

        // Always draw center point first (even if box is tiny)
        drawCircle(
            color = Color.White,
            radius = 2.dp.toPx(),
            center = Offset(centerX, centerY)
        )

        Log.d("DrawDetections", "Detection $index: Drawing center at: ($centerX, $centerY)")

        // Only draw bounding box if it has reasonable dimensions
        if (right > left + 10f && bottom > top + 10f) {
            // Draw bounding box with very thick stroke
            drawRect(
                color = Color.Green,
                topLeft = Offset(left, top),
                size = Size(right - left, bottom - top),
                style = Stroke(width = 1.dp.toPx())
            )

            // Draw confidence score with background
            val label = "${detector.getClassLabel(detection.classId)}: ${String.format("%.2f", detection.score)}"
            val textSize = 18.sp.toPx()
            val textPadding = 8.dp.toPx()

            // Calculate label position (prefer above box, but use below if near top)
            val labelTop = if (top > textSize + textPadding * 2f) {
                top - textSize - textPadding * 2f
            } else {
                bottom + textPadding
            }

            // Draw label text
            drawContext.canvas.nativeCanvas.apply {
                val paint = android.graphics.Paint().apply {
                    color = Color.Green.toArgb()
                    this.textSize = 22.sp.toPx()
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
            Log.w("DrawDetections", "Detection $index: Box too small or invalid - left=$left, top=$top, right=$right, bottom=$bottom")

            // Draw a large circle around tiny detections
            drawCircle(
                color = Color.Green,
                radius = 40.dp.toPx(),
                center = Offset(centerX, centerY),
                style = Stroke(width = 6.dp.toPx())
            )
        }
    }
}

private fun DrawScope.drawBarPaths(paths: List<BarPath>) {
    val canvasWidth = size.width
    val canvasHeight = size.height

    Log.d("DrawBarPaths", "Drawing ${paths.size} paths, canvas: ${canvasWidth}x${canvasHeight}")

    paths.forEachIndexed { pathIndex, path ->
        val points = path.points
        Log.d("DrawBarPaths", "Path $pathIndex has ${points.size} points")

        if (points.size > 1) {
            // Draw path line with thicker stroke
            for (i in 0 until points.size - 1) {
                val startPoint = points[i]
                val endPoint = points[i + 1]

                val startX = startPoint.x * canvasWidth
                val startY = startPoint.y * canvasHeight
                val endX = endPoint.x * canvasWidth
                val endY = endPoint.y * canvasHeight

                Log.d("DrawBarPaths", "Drawing line from ($startX, $startY) to ($endX, $endY)")

                drawLine(
                    color = path.color,
                    start = Offset(startX, startY),
                    end = Offset(endX, endY),
                    strokeWidth = 5.dp.toPx(), // Thicker line
                    pathEffect = PathEffect.dashPathEffect(floatArrayOf(15f, 10f), 0f)
                )
            }

            // Draw path points with fade effect (larger points)
            points.forEachIndexed { index, point ->
                val alpha = (index.toFloat() / points.size) * 0.8f + 0.2f
                val pointX = point.x * canvasWidth
                val pointY = point.y * canvasHeight

                drawCircle(
                    color = path.color.copy(alpha = alpha),
                    radius = 8.dp.toPx(), // Larger points
                    center = Offset(pointX, pointY)
                )
            }
        } else if (points.size == 1) {
            // Draw single point if we only have one
            val point = points[0]
            val pointX = point.x * canvasWidth
            val pointY = point.y * canvasHeight

            Log.d("DrawBarPaths", "Drawing single point at ($pointX, $pointY)")

            drawCircle(
                color = path.color,
                radius = 10.dp.toPx(),
                center = Offset(pointX, pointY)
            )
        }
    }
}

// Data class for bar path processing results
data class BarPathResult(
    val paths: List<BarPath>,
    val movement: MovementAnalysis?,
    val repCount: Int
)

// Keep existing helper functions (processBarPath, analyzeMovement, countReps, etc.)
private fun processBarPath(
    detections: List<Detection>,
    currentPaths: List<BarPath>,
    maxPoints: Int,
    minThreshold: Float
): BarPathResult {
    Log.d("BarPath", "processBarPath called - detections: ${detections.size}, currentPaths: ${currentPaths.size}")

    if (detections.isEmpty()) {
        Log.d("BarPath", "No detections, returning current state")
        return BarPathResult(currentPaths, null, 0)
    }

    // Get the center point of the first (most confident) detection
    val detection = detections.first()
    val centerX = (detection.bbox.left + detection.bbox.right) / 2f
    val centerY = (detection.bbox.top + detection.bbox.bottom) / 2f
    val currentTime = System.currentTimeMillis()

    val newPoint = PathPoint(centerX, centerY, currentTime)
    Log.d("BarPath", "New point: ($centerX, $centerY) at time $currentTime")

    // Get the current active path
    val activePath = currentPaths.lastOrNull() ?: BarPath()

    // Check if this is a significant movement
    val shouldAddPoint = if (activePath.points.isEmpty()) {
        Log.d("BarPath", "First point in path")
        true
    } else {
        val lastPoint = activePath.points.last()
        val distance = newPoint.distanceTo(lastPoint)
        Log.d("BarPath", "Distance from last point: $distance, threshold: $minThreshold")
        distance > minThreshold
    }

    if (!shouldAddPoint) {
        Log.d("BarPath", "Movement too small, not adding point")
        return BarPathResult(currentPaths, null, 0)
    }

    // Add point to active path
    activePath.addPoint(newPoint, maxPoints)
    Log.d("BarPath", "Added point to path, now has ${activePath.points.size} points")

    // Update paths list
    val updatedPaths = if (currentPaths.isEmpty()) {
        listOf(activePath)
    } else {
        currentPaths.dropLast(1) + activePath
    }

    // Analyze movement
    val movement = analyzeMovement(activePath.points)

    // Count reps
    val repCount = countReps(activePath.points)

    Log.d("BarPath", "Bar path updated - Points: ${activePath.points.size}, Reps: $repCount")

    return BarPathResult(updatedPaths, movement, repCount)
}

// Function to analyze movement direction and velocity
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

// Function to count reps based on direction changes
private fun countReps(points: List<PathPoint>): Int {
    if (points.size < 20) return 0

    var repCount = 0
    var lastDirection: MovementDirection? = null
    var inUpPhase = false
    val smoothingWindow = 5
    val repThreshold = 0.05f // Minimum vertical displacement for a rep

    for (i in smoothingWindow until points.size - smoothingWindow) {
        val beforeY = points.subList(i - smoothingWindow, i).map { it.y }.average().toFloat()
        val afterY = points.subList(i, i + smoothingWindow).map { it.y }.average().toFloat()

        val currentDirection = when {
            afterY - beforeY > 0.02f -> MovementDirection.DOWN
            afterY - beforeY < -0.02f -> MovementDirection.UP
            else -> MovementDirection.STABLE
        }

        // Detect rep completion: UP -> DOWN transition with sufficient displacement
        if (lastDirection == MovementDirection.UP && currentDirection == MovementDirection.DOWN && inUpPhase) {
            val upStartIndex = findLastDirectionChange(points, i, MovementDirection.DOWN, MovementDirection.UP, smoothingWindow)
            if (upStartIndex != -1) {
                val displacement = abs(points[i].y - points[upStartIndex].y)
                if (displacement > repThreshold) {
                    repCount++
                    inUpPhase = false
                    Log.d("RepCounter", "Rep detected! Count: $repCount, Displacement: $displacement")
                }
            }
        }

        // Start tracking up phase
        if (lastDirection == MovementDirection.DOWN && currentDirection == MovementDirection.UP) {
            inUpPhase = true
        }

        if (currentDirection != MovementDirection.STABLE) {
            lastDirection = currentDirection
        }
    }

    return repCount
}

// Helper function to find the last direction change
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
private fun EnhancedInfoPanel(
    detections: List<Detection>,
    fps: Float,
    isProcessing: Boolean,
    currentMovement: MovementAnalysis?,
    repCount: Int,
    isRecording: Boolean,
    isGeneratingReport: Boolean,
    onStartStopRecording: () -> Unit,
    onClearPath: () -> Unit,
    onGenerateExcelReport: () -> Unit,
    onGenerateCSVReport: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .padding(top = 60.dp)
    ) {
        Column(
            modifier = Modifier.align(Alignment.TopCenter),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "Bar Path Detector Pro",
                color = Color.White,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )
            Spacer(modifier = Modifier.height(4.dp))

            // Control buttons row 1
            Row(
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.padding(horizontal = 8.dp)
            ) {
                Button(
                    onClick = onStartStopRecording,
                    modifier = Modifier.height(28.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (isRecording) Color.Red else Color.Green
                    )
                ) {
                    Text(
                        text = if (isRecording) "Stop" else "Start",
                        fontSize = 10.sp,
                        color = Color.White
                    )
                }
                Button(
                    onClick = onClearPath,
                    modifier = Modifier.height(28.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color.Blue)
                ) {
                    Text("Clear", fontSize = 10.sp, color = Color.White)
                }
            }

            Spacer(modifier = Modifier.height(4.dp))

            // Report generation buttons row 2
            AnimatedVisibility(visible = repCount > 0) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    modifier = Modifier.padding(horizontal = 8.dp)
                ) {
                    Button(
                        onClick = onGenerateExcelReport,
                        enabled = !isGeneratingReport && repCount > 0,
                        modifier = Modifier.height(28.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF007ACC),
                            disabledContainerColor = Color.Gray
                        )
                    ) {
                        Text(
                            text = if (isGeneratingReport) "..." else "📊 Excel",
                            fontSize = 10.sp,
                            color = Color.White
                        )
                    }
                    Button(
                        onClick = onGenerateCSVReport,
                        enabled = !isGeneratingReport && repCount > 0,
                        modifier = Modifier.height(28.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF228B22),
                            disabledContainerColor = Color.Gray
                        )
                    ) {
                        Text(
                            text = if (isGeneratingReport) "..." else "📋 CSV",
                            fontSize = 10.sp,
                            color = Color.White
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(6.dp))

            // Status info
            Text(
                text = "FPS: ${String.format("%.1f", fps)}",
                color = Color.White,
                fontSize = 11.sp,
                textAlign = TextAlign.Center
            )
            Text(
                text = "Detections: ${detections.size}",
                color = Color.White,
                fontSize = 11.sp,
                textAlign = TextAlign.Center
            )
            Text(
                text = "Recording: ${if (isRecording) "ON" else "OFF"}",
                color = if (isRecording) Color.Green else Color.Gray,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )
            Text(
                text = "Reps: $repCount",
                color = Color.Cyan,
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )

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
                Text(
                    text = "Movement: ${movement.direction}",
                    color = when (movement.direction) {
                        MovementDirection.UP -> Color.Green
                        MovementDirection.DOWN -> Color.Red
                        MovementDirection.STABLE -> Color.Yellow
                    },
                    fontSize = 11.sp,
                    textAlign = TextAlign.Center
                )
                Text(
                    text = "Velocity: ${String.format("%.2f", movement.velocity)}",
                    color = Color.White,
                    fontSize = 9.sp,
                    textAlign = TextAlign.Center
                )
                Text(
                    text = "Distance: ${String.format("%.2f", movement.totalDistance)}",
                    color = Color.White,
                    fontSize = 9.sp,
                    textAlign = TextAlign.Center
                )
            }

            if (detections.isNotEmpty()) {
                Spacer(modifier = Modifier.height(6.dp))
                detections.take(2).forEachIndexed { index, detection ->
                    Text(
                        text = "Barbell ${index + 1}: ${String.format("%.2f", detection.score)}",
                        color = Color.Cyan,
                        fontSize = 9.sp,
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
    }
}