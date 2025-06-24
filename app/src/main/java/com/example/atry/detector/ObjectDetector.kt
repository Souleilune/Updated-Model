package com.example.atry.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Enhanced YOLOv11 Object Detector with tracking support for bar path analysis
 */
class YOLOv11ObjectDetector(
    context: Context,
    modelPath: String = "new model.tflite",
    private val inputSize: Int = 640,
    private val confThreshold: Float = 0.4f,
    private val iouThreshold: Float = 0.3f,
    private val maxDetections: Int = 20
) {

    // Class labels - since you only have "Barbell" as class 0
    private val classLabels = arrayOf("Barbell")

    private val interpreter: Interpreter

    // YOLOv11 outputs: Let's detect the actual output shape
    private lateinit var outputBuffer: Array<Array<FloatArray>>
    private var actualOutputShape: IntArray = intArrayOf()

    // Tracking state for better bar path detection
    private var previousDetections = mutableListOf<TrackedDetection>()
    private var frameCounter = 0

    companion object {
        private const val TAG = "YOLOv11Detector"
    }

    init {
        val options = Interpreter.Options()
        options.setNumThreads(4) // Use multiple CPU threads for better performance

        interpreter = Interpreter(loadModelFile(context, modelPath), options)

        // Log model input/output info and initialize buffers
        initializeModelBuffers()

        // Debug model information
        DebugUtils.debugModelInfo(interpreter)

        // Test with synthetic image
        DebugUtils.testModelWithSyntheticImage(this, context)
    }

    private fun loadModelFile(context: Context, assetPath: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(assetPath)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun initializeModelBuffers() {
        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)

        actualOutputShape = outputTensor.shape()

        Log.d(TAG, "YOLOv11 Model Info:")
        Log.d(TAG, "Input shape: ${inputTensor.shape().contentToString()}")
        Log.d(TAG, "Input data type: ${inputTensor.dataType()}")
        Log.d(TAG, "Output shape: ${actualOutputShape.contentToString()}")
        Log.d(TAG, "Output data type: ${outputTensor.dataType()}")

        // Initialize output buffer based on actual shape
        when (actualOutputShape.size) {
            3 -> {
                // Shape: [1, features, detections] - most common for YOLOv11
                val batchSize = actualOutputShape[0]
                val features = actualOutputShape[1]
                val detections = actualOutputShape[2]
                outputBuffer = Array(batchSize) { Array(features) { FloatArray(detections) } }
                Log.d(TAG, "Initialized 3D output buffer: [$batchSize, $features, $detections]")
            }
            2 -> {
                // Shape: [detections, features] - flattened format
                val detections = actualOutputShape[0]
                val features = actualOutputShape[1]
                outputBuffer = Array(1) { Array(detections) { FloatArray(features) } }
                Log.d(TAG, "Initialized 2D output buffer: [1, $detections, $features]")
            }
            else -> {
                Log.e(TAG, "Unexpected output shape: ${actualOutputShape.contentToString()}")
                // Fallback to default
                outputBuffer = Array(1) { Array(5) { FloatArray(8400) } }
            }
        }
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Resize image to model input size
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // Allocate ByteBuffer for model input
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        // Convert bitmap to normalized float values
        val intValues = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        var pixelIndex = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues[pixelIndex++]
                // Normalize RGB values to [0, 1]
                byteBuffer.putFloat(((pixelValue shr 16) and 0xFF) / 255.0f)
                byteBuffer.putFloat(((pixelValue shr 8) and 0xFF) / 255.0f)
                byteBuffer.putFloat((pixelValue and 0xFF) / 255.0f)
            }
        }

        byteBuffer.rewind()
        return byteBuffer
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        frameCounter++

        try {
            // Preprocess input
            val inputBuffer = preprocessImage(bitmap)

            // Run inference
            interpreter.run(inputBuffer, outputBuffer)

            // Post-process results with tracking
            return postProcessWithTracking(bitmap.width, bitmap.height)
        } catch (e: Exception) {
            Log.e(TAG, "Error during detection: ${e.message}", e)
            return emptyList()
        }
    }

    private fun postProcessWithTracking(originalWidth: Int, originalHeight: Int): List<Detection> {
        val rawDetections = postProcess(originalWidth, originalHeight)

        Log.d(TAG, "Raw detections found: ${rawDetections.size}")

        // Debug raw model output occasionally
        if (frameCounter % 30 == 0) { // Every 30 frames
            DebugUtils.logRawModelOutput(outputBuffer, maxSamples = 5)
        }

        // Apply temporal filtering to reduce jitter
        val filteredDetections = applyTemporalFiltering(rawDetections)

        // Update tracking state
        updateTracking(filteredDetections)

        return filteredDetections
    }

    // Replace these functions in your ObjectDetector.kt file:

    private fun postProcess(originalWidth: Int, originalHeight: Int): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            when (actualOutputShape.size) {
                3 -> {
                    // Standard YOLOv11 format: [1, features, detections]
                    val features = actualOutputShape[1]
                    val numDetections = actualOutputShape[2]

                    Log.d(TAG, "Processing 3D output: features=$features, detections=$numDetections")

                    if (features >= 5) {
                        // Extract predictions
                        val centerXArray = outputBuffer[0][0] // x coordinates
                        val centerYArray = outputBuffer[0][1] // y coordinates
                        val widthArray = outputBuffer[0][2]   // width values
                        val heightArray = outputBuffer[0][3]  // height values
                        val confidenceArray = outputBuffer[0][4] // confidence scores

                        for (i in 0 until numDetections) {
                            val centerX = centerXArray[i]
                            val centerY = centerYArray[i]
                            val width = widthArray[i]
                            val height = heightArray[i]
                            val confidence = confidenceArray[i]

                            // FIXED: Process detection with correct coordinate handling
                            processDetectionFixed(centerX, centerY, width, height, confidence, detections)
                        }
                    }
                }
                2 -> {
                    // Flattened format: [detections, features]
                    val numDetections = actualOutputShape[0]
                    val features = actualOutputShape[1]

                    Log.d(TAG, "Processing 2D output: detections=$numDetections, features=$features")

                    if (features >= 5) {
                        for (i in 0 until numDetections) {
                            val detection = outputBuffer[0][i]
                            val centerX = detection[0]
                            val centerY = detection[1]
                            val width = detection[2]
                            val height = detection[3]
                            val confidence = detection[4]

                            // FIXED: Process detection with correct coordinate handling
                            processDetectionFixed(centerX, centerY, width, height, confidence, detections)
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in postProcess: ${e.message}", e)
        }

        Log.d(TAG, "Detections after confidence filtering: ${detections.size}")

        // Apply Non-Maximum Suppression
        val finalDetections = applyNMS(detections)
        Log.d(TAG, "Final detections after NMS: ${finalDetections.size}")

        return finalDetections
    }

    // NEW FIXED FUNCTION: Proper coordinate handling for YOLOv11
    private fun processDetectionFixed(
        centerX: Float,
        centerY: Float,
        width: Float,
        height: Float,
        confidence: Float,
        detections: MutableList<Detection>
    ) {
        // Filter by confidence threshold
        if (confidence >= confThreshold) {
            Log.d(TAG, "Detection found - conf: $confidence, center: ($centerX, $centerY), size: (${width}x$height)")

            // YOLOv11 outputs are ALREADY normalized to [0,1] range
            // No need to divide by inputSize again!
            val normalizedCenterX = centerX.coerceIn(0f, 1f)
            val normalizedCenterY = centerY.coerceIn(0f, 1f)
            val normalizedWidth = width.coerceIn(0f, 1f)
            val normalizedHeight = height.coerceIn(0f, 1f)

            Log.d(TAG, "Using coordinates as-is: center=($normalizedCenterX, $normalizedCenterY), size=(${normalizedWidth}x$normalizedHeight)")

            // Convert center coordinates to corner coordinates
            val left = (normalizedCenterX - normalizedWidth / 2f).coerceIn(0f, 1f)
            val top = (normalizedCenterY - normalizedHeight / 2f).coerceIn(0f, 1f)
            val right = (normalizedCenterX + normalizedWidth / 2f).coerceIn(0f, 1f)
            val bottom = (normalizedCenterY + normalizedHeight / 2f).coerceIn(0f, 1f)

            Log.d(TAG, "Final normalized bbox - left: $left, top: $top, right: $right, bottom: $bottom")

            // Validate bounding box
            if (right > left && bottom > top) {
                val bbox = RectF(left, top, right, bottom)
                val detection = Detection(bbox, confidence, 0) // Class 0 = Barbell

                // Validate detection before adding
                if (DebugUtils.validateDetection(detection)) {
                    detections.add(detection)
                    Log.d(TAG, "✅ Added valid detection with bbox: $bbox")
                } else {
                    Log.w(TAG, "❌ Invalid detection failed validation")
                }
            } else {
                Log.w(TAG, "❌ Invalid bounding box dimensions - left: $left, top: $top, right: $right, bottom: $bottom")
            }
        }
    }

    private fun applyTemporalFiltering(detections: List<Detection>): List<Detection> {
        if (previousDetections.isEmpty() || detections.isEmpty()) {
            return detections
        }

        val filteredDetections = mutableListOf<Detection>()

        for (detection in detections) {
            // Find the closest previous detection
            val closestPrevious = findClosestPreviousDetection(detection)

            if (closestPrevious != null) {
                // Apply smoothing to reduce jitter
                val smoothingFactor = 0.3f // Reduced for more responsiveness
                val smoothedBbox = RectF(
                    detection.bbox.left * (1 - smoothingFactor) + closestPrevious.detection.bbox.left * smoothingFactor,
                    detection.bbox.top * (1 - smoothingFactor) + closestPrevious.detection.bbox.top * smoothingFactor,
                    detection.bbox.right * (1 - smoothingFactor) + closestPrevious.detection.bbox.right * smoothingFactor,
                    detection.bbox.bottom * (1 - smoothingFactor) + closestPrevious.detection.bbox.bottom * smoothingFactor
                )

                filteredDetections.add(Detection(smoothedBbox, detection.score, detection.classId))
            } else {
                // No previous detection found, use current detection as-is
                filteredDetections.add(detection)
            }
        }

        return filteredDetections
    }

    private fun findClosestPreviousDetection(detection: Detection): TrackedDetection? {
        if (previousDetections.isEmpty()) return null

        val detectionCenter = getCenterPoint(detection.bbox)
        var closestDetection: TrackedDetection? = null
        var minDistance = Float.MAX_VALUE

        for (prevDetection in previousDetections) {
            val prevCenter = getCenterPoint(prevDetection.detection.bbox)
            val distance = calculateDistance(detectionCenter, prevCenter)

            if (distance < minDistance && distance < 0.15f) { // Increased threshold
                minDistance = distance
                closestDetection = prevDetection
            }
        }

        return closestDetection
    }

    private fun updateTracking(detections: List<Detection>) {
        // Update previous detections for next frame
        previousDetections.clear()
        detections.forEach { detection ->
            previousDetections.add(TrackedDetection(detection, frameCounter))
        }

        // Remove old tracking data (older than 10 frames)
        previousDetections.removeAll { frameCounter - it.frameNumber > 10 }
    }

    private fun getCenterPoint(bbox: RectF): Pair<Float, Float> {
        return Pair(
            (bbox.left + bbox.right) / 2f,
            (bbox.top + bbox.bottom) / 2f
        )
    }

    private fun calculateDistance(point1: Pair<Float, Float>, point2: Pair<Float, Float>): Float {
        val dx = point1.first - point2.first
        val dy = point1.second - point2.second
        return sqrt(dx * dx + dy * dy)
    }

    private fun applyNMS(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        val sortedDetections = detections.sortedByDescending { it.score }.toMutableList()
        val finalDetections = mutableListOf<Detection>()

        while (sortedDetections.isNotEmpty() && finalDetections.size < maxDetections) {
            val bestDetection = sortedDetections.removeAt(0)
            finalDetections.add(bestDetection)

            // Remove overlapping detections
            val iterator = sortedDetections.iterator()
            while (iterator.hasNext()) {
                val detection = iterator.next()
                if (calculateIoU(bestDetection.bbox, detection.bbox) > iouThreshold) {
                    iterator.remove()
                }
            }
        }

        return finalDetections
    }

    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intersectionLeft = max(box1.left, box2.left)
        val intersectionTop = max(box1.top, box2.top)
        val intersectionRight = min(box1.right, box2.right)
        val intersectionBottom = min(box1.bottom, box2.bottom)

        val intersectionWidth = max(0f, intersectionRight - intersectionLeft)
        val intersectionHeight = max(0f, intersectionBottom - intersectionTop)
        val intersectionArea = intersectionWidth * intersectionHeight

        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
        val unionArea = box1Area + box2Area - intersectionArea

        return if (unionArea > 0) intersectionArea / unionArea else 0f
    }

    fun getClassLabel(classId: Int): String {
        return if (classId < classLabels.size) classLabels[classId] else "Unknown"
    }

    fun getDetectionCenter(detection: Detection): Pair<Float, Float> {
        return getCenterPoint(detection.bbox)
    }

    fun getDetectionQuality(detection: Detection): DetectionQuality {
        val bbox = detection.bbox
        val width = bbox.right - bbox.left
        val height = bbox.bottom - bbox.top
        val area = width * height
        val aspectRatio = width / height

        return DetectionQuality(
            confidence = detection.score,
            size = area,
            aspectRatio = aspectRatio,
            stability = calculateStability(detection)
        )
    }

    private fun calculateStability(detection: Detection): Float {
        // Calculate how stable this detection is based on previous frames
        val closestPrevious = findClosestPreviousDetection(detection)
        return if (closestPrevious != null) {
            val distance = calculateDistance(
                getCenterPoint(detection.bbox),
                getCenterPoint(closestPrevious.detection.bbox)
            )
            maxOf(0f, 1f - distance * 10f) // Higher stability for smaller movement
        } else {
            0.5f // Neutral stability for new detections
        }
    }

    fun close() {
        interpreter.close()
    }
}

/**
 * Data class for tracking detections across frames
 */
data class TrackedDetection(
    val detection: Detection,
    val frameNumber: Int,
    val trackId: Int = generateTrackId()
) {
    companion object {
        private var trackCounter = 0
        private fun generateTrackId(): Int = ++trackCounter
    }
}

/**
 * Data class for detection quality metrics
 */
data class DetectionQuality(
    val confidence: Float,
    val size: Float,
    val aspectRatio: Float,
    val stability: Float
) {
    fun getOverallQuality(): Float {
        return (confidence * 0.4f +
                minOf(size * 10f, 1f) * 0.2f +
                (1f - kotlin.math.abs(aspectRatio - 1f)) * 0.2f +
                stability * 0.2f)
    }
}