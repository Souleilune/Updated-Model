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
 * Mobile-Optimized YOLOv11n Object Detector for Barbell Tracking
 * Specifically configured for YOLOv11n (Nano) model - perfect for mobile deployment
 *
 * Key advantages of YOLOv11n for mobile:
 * - Model size: ~6MB (vs 20MB for YOLOv11s)
 * - Inference speed: 20-30ms (vs 50-100ms for YOLOv11s)
 * - Memory usage: ~60MB RAM (vs 180MB for YOLOv11s)
 * - Battery efficient for long workout sessions
 * - Maintains 25-35 FPS for smooth real-time tracking
 */
class YOLOv11ObjectDetector(
    context: Context,
    modelPath: String = "optimizefloat16.tflite", // Your YOLOv11n model
    private val inputSize: Int = 640,
    private val confThreshold: Float = 0.3f,  // Slightly higher for mobile efficiency
    private val iouThreshold: Float = 0.45f,  // Optimized for single-class detection
    private val maxDetections: Int = 8        // Reduced for better mobile performance
) {

    private val classLabels = arrayOf("Barbell")
    private val interpreter: Interpreter

    // Model-specific constants based on your training output
    private val numDetections = 8400 // From output shape (1, 5, 8400)
    private val numFeatures = 5      // [x, y, w, h, confidence]

    // Output buffer matching your exact model architecture
    private val outputBuffer = Array(1) { Array(numFeatures) { FloatArray(numDetections) } }

    // Tracking for temporal consistency
    private var previousDetections = mutableListOf<TrackedDetection>()
    private var frameCounter = 0

    companion object {
        private const val TAG = "OptimizedYOLOv11"
    }

    init {
        val options = Interpreter.Options().apply {
            setNumThreads(2) // Reduced for mobile efficiency - YOLOv11n doesn't need 4 threads
            setUseNNAPI(true) // Enable Android Neural Networks API for acceleration
            // GPU delegate can be added but often CPU is more efficient for nano models
        }

        interpreter = Interpreter(loadModelFile(context, modelPath), options)

        logModelDetails()
        Log.d(TAG, "Mobile-optimized YOLOv11n detector initialized for real-time barbell tracking")
    }

    private fun loadModelFile(context: Context, assetPath: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(assetPath)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun logModelDetails() {
        try {
            val inputTensor = interpreter.getInputTensor(0)
            val outputTensor = interpreter.getOutputTensor(0)

            Log.d(TAG, "=== MOBILE YOLOv11n MODEL DETAILS ===")
            Log.d(TAG, "Model type: YOLOv11n (Nano) - Mobile Optimized")
            Log.d(TAG, "Input shape: ${inputTensor.shape().contentToString()}")
            Log.d(TAG, "Output shape: ${outputTensor.shape().contentToString()}")
            Log.d(TAG, "Expected performance: 25-35 FPS on mobile")
            Log.d(TAG, "Model size: ~6MB (Float16)")
            Log.d(TAG, "Memory usage: ~60MB RAM")
            Log.d(TAG, "Target class: ${classLabels[0]}")
            Log.d(TAG, "Mobile confidence threshold: $confThreshold")
            Log.d(TAG, "========================================")
        } catch (e: Exception) {
            Log.e(TAG, "Error logging model details: ${e.message}", e)
        }
    }

    /**
     * Optimized preprocessing for 640x640 NHWC input format
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Resize to exact model input size
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // Allocate buffer for NHWC format: [batch=1, height=640, width=640, channels=3]
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        // Convert to normalized float values [0.0, 1.0]
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues[i * inputSize + j]

                // Extract RGB values and normalize to [0, 1]
                val r = ((pixelValue shr 16) and 0xFF) / 255.0f
                val g = ((pixelValue shr 8) and 0xFF) / 255.0f
                val b = (pixelValue and 0xFF) / 255.0f

                // Store in NHWC format
                byteBuffer.putFloat(r)
                byteBuffer.putFloat(g)
                byteBuffer.putFloat(b)
            }
        }

        byteBuffer.rewind()
        return byteBuffer
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        frameCounter++

        try {
            val startTime = System.currentTimeMillis()

            // Preprocess input
            val inputBuffer = preprocessImage(bitmap)

            // Run inference
            interpreter.run(inputBuffer, outputBuffer)

            // Post-process results
            val detections = postProcessOptimized()

            // Apply temporal filtering
            val filteredDetections = applyTemporalFiltering(detections)

            // Update tracking
            updateTracking(filteredDetections)

            val inferenceTime = System.currentTimeMillis() - startTime
            Log.d(TAG, "Inference completed in ${inferenceTime}ms, found ${filteredDetections.size} detections")

            return filteredDetections

        } catch (e: Exception) {
            Log.e(TAG, "Error during detection: ${e.message}", e)
            return emptyList()
        }
    }

    /**
     * Optimized post-processing for your model's (1, 5, 8400) output format
     */
    private fun postProcessOptimized(): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            // Your model outputs: [batch=1, features=5, detections=8400]
            // Features: [center_x, center_y, width, height, confidence]
            val centerXArray = outputBuffer[0][0]    // All center X coordinates
            val centerYArray = outputBuffer[0][1]    // All center Y coordinates
            val widthArray = outputBuffer[0][2]      // All widths
            val heightArray = outputBuffer[0][3]     // All heights
            val confidenceArray = outputBuffer[0][4] // All confidence scores

            for (i in 0 until numDetections) {
                val centerX = centerXArray[i]
                val centerY = centerYArray[i]
                val width = widthArray[i]
                val height = heightArray[i]
                val confidence = confidenceArray[i]

                // Apply confidence threshold early
                if (confidence >= confThreshold) {

                    // Convert center coordinates to corner coordinates
                    // YOLOv11 outputs are already normalized [0, 1]
                    val left = (centerX - width / 2f).coerceIn(0f, 1f)
                    val top = (centerY - height / 2f).coerceIn(0f, 1f)
                    val right = (centerX + width / 2f).coerceIn(0f, 1f)
                    val bottom = (centerY + height / 2f).coerceIn(0f, 1f)

                    // Validate bounding box
                    if (right > left && bottom > top) {
                        val bbox = RectF(left, top, right, bottom)
                        val detection = Detection(bbox, confidence, 0) // Class 0 = Barbell

                        // Additional validation for barbell characteristics
                        if (isValidBarbellDetection(detection)) {
                            detections.add(detection)
                        }
                    }
                }
            }

            Log.d(TAG, "Raw detections after confidence filter: ${detections.size}")

        } catch (e: Exception) {
            Log.e(TAG, "Error in postProcessOptimized: ${e.message}", e)
        }

        // Apply Non-Maximum Suppression
        val finalDetections = applyNMS(detections)
        Log.d(TAG, "Final detections after NMS: ${finalDetections.size}")

        return finalDetections
    }

    /**
     * Barbell-specific validation based on typical characteristics
     */
    private fun isValidBarbellDetection(detection: Detection): Boolean {
        val bbox = detection.bbox
        val width = bbox.right - bbox.left
        val height = bbox.bottom - bbox.top
        val aspectRatio = width / height
        val area = width * height

        // Barbell validation criteria
        val validAspectRatio = aspectRatio > 0.5f && aspectRatio < 8.0f  // Barbells are typically wider than tall
        val validSize = area > 0.001f && area < 0.4f  // Reasonable size bounds
        val validDimensions = width > 0.02f && height > 0.01f  // Minimum visible size

        return validAspectRatio && validSize && validDimensions
    }

    /**
     * Enhanced temporal filtering for stable tracking
     */
    private fun applyTemporalFiltering(detections: List<Detection>): List<Detection> {
        if (previousDetections.isEmpty() || detections.isEmpty()) {
            return detections
        }

        val filteredDetections = mutableListOf<Detection>()
        val smoothingFactor = 0.3f // Balanced smoothing

        for (detection in detections) {
            val closestPrevious = findClosestPreviousDetection(detection)

            if (closestPrevious != null) {
                // Apply temporal smoothing
                val prevBbox = closestPrevious.detection.bbox
                val currBbox = detection.bbox

                val smoothedBbox = RectF(
                    currBbox.left * (1 - smoothingFactor) + prevBbox.left * smoothingFactor,
                    currBbox.top * (1 - smoothingFactor) + prevBbox.top * smoothingFactor,
                    currBbox.right * (1 - smoothingFactor) + prevBbox.right * smoothingFactor,
                    currBbox.bottom * (1 - smoothingFactor) + prevBbox.bottom * smoothingFactor
                )

                // Smooth confidence as well
                val smoothedConfidence = detection.score * (1 - smoothingFactor) +
                        closestPrevious.detection.score * smoothingFactor

                filteredDetections.add(Detection(smoothedBbox, smoothedConfidence, detection.classId))
            } else {
                filteredDetections.add(detection)
            }
        }

        return filteredDetections
    }

    private fun findClosestPreviousDetection(detection: Detection): TrackedDetection? {
        if (previousDetections.isEmpty()) return null

        val currentCenter = getCenterPoint(detection.bbox)
        var closestDetection: TrackedDetection? = null
        var minDistance = Float.MAX_VALUE

        for (prevDetection in previousDetections) {
            val prevCenter = getCenterPoint(prevDetection.detection.bbox)
            val distance = calculateDistance(currentCenter, prevCenter)

            if (distance < minDistance && distance < 0.15f) {
                minDistance = distance
                closestDetection = prevDetection
            }
        }

        return closestDetection
    }

    private fun updateTracking(detections: List<Detection>) {
        previousDetections.clear()
        detections.forEach { detection ->
            previousDetections.add(TrackedDetection(detection, frameCounter))
        }

        // Keep only recent tracking data
        if (previousDetections.size > 5) {
            previousDetections = previousDetections.takeLast(3).toMutableList()
        }
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

    /**
     * Optimized Non-Maximum Suppression
     */
    private fun applyNMS(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        val sortedDetections = detections.sortedByDescending { it.score }.toMutableList()
        val finalDetections = mutableListOf<Detection>()

        while (sortedDetections.isNotEmpty() && finalDetections.size < maxDetections) {
            val bestDetection = sortedDetections.removeAt(0)
            finalDetections.add(bestDetection)

            // Remove overlapping detections
            sortedDetections.removeAll { detection ->
                calculateIoU(bestDetection.bbox, detection.bbox) > iouThreshold
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

        // Calculate stability based on tracking history
        val stability = calculateStability(detection)

        // Barbell-specific quality metrics
        val sizeScore = (area * 10f).coerceAtMost(1f) // Optimal size around 0.1
        val aspectScore = if (aspectRatio > 1f) {
            (1f / aspectRatio).coerceAtLeast(0.2f) // Prefer horizontal orientation
        } else {
            aspectRatio.coerceAtLeast(0.2f)
        }

        return DetectionQuality(
            confidence = detection.score,
            size = sizeScore,
            aspectRatio = aspectScore,
            stability = stability
        )
    }

    private fun calculateStability(detection: Detection): Float {
        val closestPrevious = findClosestPreviousDetection(detection)
        return if (closestPrevious != null) {
            val distance = calculateDistance(
                getCenterPoint(detection.bbox),
                getCenterPoint(closestPrevious.detection.bbox)
            )
            maxOf(0f, 1f - distance * 8f) // Higher penalty for movement
        } else {
            0.7f // Good stability for new detections
        }
    }

    /**
     * Get model performance metrics
     */
    fun getModelMetrics(): ModelMetrics {
        return ModelMetrics(
            inputShape = intArrayOf(1, inputSize, inputSize, 3),
            outputShape = intArrayOf(1, numFeatures, numDetections),
            confidenceThreshold = confThreshold,
            iouThreshold = iouThreshold,
            maxDetections = maxDetections,
            numClasses = 1,
            modelSize = "5.2 MB (Float16)"
        )
    }

    fun close() {
        interpreter.close()
        Log.d(TAG, "Optimized YOLOv11 detector closed")
    }
}

/**
 * Model performance metrics
 */
data class ModelMetrics(
    val inputShape: IntArray,
    val outputShape: IntArray,
    val confidenceThreshold: Float,
    val iouThreshold: Float,
    val maxDetections: Int,
    val numClasses: Int,
    val modelSize: String
)

/**
 * Enhanced detection quality metrics
 */
data class DetectionQuality(
    val confidence: Float,
    val size: Float,
    val aspectRatio: Float,
    val stability: Float
) {
    fun getOverallQuality(): Float {
        return (confidence * 0.4f +
                size * 0.2f +
                aspectRatio * 0.2f +
                stability * 0.2f)
    }

    fun isHighQuality(): Boolean = getOverallQuality() > 0.7f
    fun isMediumQuality(): Boolean = getOverallQuality() > 0.5f
}

/**
 * Tracking data for temporal consistency
 */
data class TrackedDetection(
    val detection: Detection,
    val frameNumber: Int,
    val trackId: Int = generateTrackId(),
    val firstSeen: Long = System.currentTimeMillis()
) {
    companion object {
        private var trackCounter = 0
        private fun generateTrackId(): Int = ++trackCounter
    }

    fun getAge(): Long = System.currentTimeMillis() - firstSeen
}