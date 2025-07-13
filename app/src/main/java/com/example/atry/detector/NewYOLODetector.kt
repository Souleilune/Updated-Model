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
 * BRAND NEW YOLOv11 Detector - Fixed Input Size Issue
 * This completely replaces the old detector with proper tensor sizing
 */
class NewYOLODetector(
    context: Context,
    modelPath: String = "optimizefloat16.tflite",
    private val inputSize: Int = 640,
    private val confThreshold: Float = 0.05f,
    private val iouThreshold: Float = 0.45f,
    private val maxDetections: Int = 10
) {

    private val classLabels = arrayOf("Barbell")
    private val interpreter: Interpreter

    // Model-specific constants
    private val numDetections = 8400
    private val numFeatures = 5

    // Output buffer matching your exact model architecture
    private val outputBuffer = Array(1) { Array(numFeatures) { FloatArray(numDetections) } }

    // Tracking for temporal consistency
    private var previousDetections = mutableListOf<TrackedDetection>()
    private var frameCounter = 0

    companion object {
        private const val TAG = "NewYOLODetector"
    }

    init {
        val options = Interpreter.Options().apply {
            setNumThreads(3)
            setUseNNAPI(false)
        }

        interpreter = Interpreter(loadModelFile(context, modelPath), options)
        logModelDetails()
        Log.d(TAG, "✅ NEW YOLOv11 detector initialized successfully")
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

            Log.d(TAG, "=== NEW YOLO DETECTOR MODEL DETAILS ===")
            Log.d(TAG, "Model file: optimizefloat16.tflite")
            Log.d(TAG, "Input shape: ${inputTensor.shape().contentToString()}")
            Log.d(TAG, "Output shape: ${outputTensor.shape().contentToString()}")
            Log.d(TAG, "Expected buffer size: ${4 * inputSize * inputSize * 3} bytes")
            Log.d(TAG, "Confidence threshold: $confThreshold")
            Log.d(TAG, "==========================================")
        } catch (e: Exception) {
            Log.e(TAG, "Error logging model details: ${e.message}", e)
        }
    }

    /**
     * MAIN DETECTION METHOD - Fixed tensor sizing
     */
    fun detect(bitmap: Bitmap): List<Detection> {
        frameCounter++

        try {
            Log.d(TAG, "🔍 Starting detection on frame $frameCounter")

            // CORRECT preprocessing with proper input size
            val inputBuffer = preprocessImageCorrectly(bitmap)

            // Clear previous output
            for (i in 0 until 1) {
                for (j in 0 until numFeatures) {
                    for (k in 0 until numDetections) {
                        outputBuffer[i][j][k] = 0f
                    }
                }
            }

            // Run inference
            val startTime = System.currentTimeMillis()
            interpreter.run(inputBuffer, outputBuffer)
            val inferenceTime = System.currentTimeMillis() - startTime

            Log.d(TAG, "✅ Inference completed in ${inferenceTime}ms")

            // Process results
            val rawDetections = postProcessResults()
            Log.d(TAG, "Found ${rawDetections.size} raw detections")

            // Apply temporal filtering occasionally
            val filteredDetections = if (frameCounter % 3 == 0) {
                applyTemporalFiltering(rawDetections)
            } else {
                rawDetections
            }

            updateTracking(filteredDetections)
            Log.d(TAG, "✅ Final detections: ${filteredDetections.size}")

            return filteredDetections

        } catch (e: Exception) {
            Log.e(TAG, "❌ Error during detection: ${e.message}", e)
            return emptyList()
        }
    }

    /**
     * FIXED preprocessing - ensures correct tensor size
     */
    private fun preprocessImageCorrectly(bitmap: Bitmap): ByteBuffer {
        Log.d(TAG, "Preprocessing: ${bitmap.width}x${bitmap.height} -> ${inputSize}x${inputSize}")

        // CRITICAL: Resize to exact model input size (640x640)
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // CRITICAL: Allocate buffer with EXACT size expected by model
        val bufferSize = 4 * inputSize * inputSize * 3  // 4,915,200 bytes for 640x640x3 float32
        val byteBuffer = ByteBuffer.allocateDirect(bufferSize)
        byteBuffer.order(ByteOrder.nativeOrder())

        Log.d(TAG, "✅ ByteBuffer allocated: $bufferSize bytes")

        val intValues = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        // Normalize to [0.0, 1.0] for YOLOv11
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val pixelValue = intValues[i * inputSize + j]

                val r = ((pixelValue shr 16) and 0xFF) / 255.0f
                val g = ((pixelValue shr 8) and 0xFF) / 255.0f
                val b = (pixelValue and 0xFF) / 255.0f

                byteBuffer.putFloat(r)
                byteBuffer.putFloat(g)
                byteBuffer.putFloat(b)
            }
        }

        byteBuffer.rewind()
        Log.d(TAG, "✅ Preprocessing completed. Buffer ready: ${byteBuffer.limit()} bytes")
        return byteBuffer
    }

    /**
     * Process model outputs
     */
    private fun postProcessResults(): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            Log.d(TAG, "Processing model outputs...")

            val centerXArray = outputBuffer[0][0]
            val centerYArray = outputBuffer[0][1]
            val widthArray = outputBuffer[0][2]
            val heightArray = outputBuffer[0][3]
            val confidenceArray = outputBuffer[0][4]

            var validCount = 0
            var confidentCount = 0

            // Debug: Check first few outputs every 10 frames
            if (frameCounter % 10 == 0) {
                Log.d(TAG, "=== MODEL OUTPUT SAMPLE ===")
                for (i in 0 until 5) {
                    Log.d(TAG, "Output[$i]: conf=${confidenceArray[i]}, x=${centerXArray[i]}, y=${centerYArray[i]}")
                }

                // Quick stats
                val maxConf = confidenceArray.maxOrNull() ?: 0f
                val avgConf = confidenceArray.take(100).average().toFloat()
                Log.d(TAG, "Max confidence: $maxConf, Avg confidence: $avgConf")
                Log.d(TAG, "===========================")
            }

            // Process all detections
            for (i in 0 until numDetections) {
                val centerX = centerXArray[i]
                val centerY = centerYArray[i]
                val width = widthArray[i]
                val height = heightArray[i]
                val confidence = confidenceArray[i]

                if (confidence > 0.01f) {
                    validCount++
                }

                if (confidence >= confThreshold) {
                    confidentCount++

                    Log.d(TAG, "🎯 CONFIDENT DETECTION! conf=$confidence")

                    // Convert to bounding box
                    val left = (centerX - width / 2f).coerceIn(0f, 1f)
                    val top = (centerY - height / 2f).coerceIn(0f, 1f)
                    val right = (centerX + width / 2f).coerceIn(0f, 1f)
                    val bottom = (centerY + height / 2f).coerceIn(0f, 1f)

                    if (right > left && bottom > top) {
                        val bbox = RectF(left, top, right, bottom)
                        val detection = Detection(bbox, confidence, 0)

                        if (isValidDetection(detection)) {
                            detections.add(detection)
                            Log.d(TAG, "✅ Valid detection added: conf=$confidence")
                        }
                    }
                }
            }

            Log.d(TAG, "Stats: valid=$validCount, confident=$confidentCount, final=${detections.size}")

        } catch (e: Exception) {
            Log.e(TAG, "Error in post-processing: ${e.message}", e)
        }

        return if (detections.isNotEmpty()) applyNMS(detections) else detections
    }

    /**
     * Very relaxed validation for testing
     */
    private fun isValidDetection(detection: Detection): Boolean {
        val bbox = detection.bbox
        val width = bbox.right - bbox.left
        val height = bbox.bottom - bbox.top
        val area = width * height

        // Very permissive validation
        val validSize = area > 0.0001f && area < 0.8f
        val validDimensions = width > 0.005f && height > 0.005f

        return validSize && validDimensions
    }

    /**
     * Non-Maximum Suppression
     */
    private fun applyNMS(detections: List<Detection>): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        val sortedDetections = detections.sortedByDescending { it.score }.toMutableList()
        val finalDetections = mutableListOf<Detection>()

        while (sortedDetections.isNotEmpty() && finalDetections.size < maxDetections) {
            val bestDetection = sortedDetections.removeAt(0)
            finalDetections.add(bestDetection)

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

    /**
     * Simple temporal filtering
     */
    private fun applyTemporalFiltering(detections: List<Detection>): List<Detection> {
        if (previousDetections.isEmpty() || detections.isEmpty()) {
            return detections
        }

        // Light smoothing for stability
        return detections.map { detection ->
            val closest = findClosestPrevious(detection)
            if (closest != null) {
                val prevBbox = closest.detection.bbox
                val currBbox = detection.bbox
                val smoothing = 0.2f

                val smoothedBbox = RectF(
                    currBbox.left * 0.8f + prevBbox.left * 0.2f,
                    currBbox.top * 0.8f + prevBbox.top * 0.2f,
                    currBbox.right * 0.8f + prevBbox.right * 0.2f,
                    currBbox.bottom * 0.8f + prevBbox.bottom * 0.2f
                )

                Detection(smoothedBbox, detection.score, detection.classId)
            } else {
                detection
            }
        }
    }

    private fun findClosestPrevious(detection: Detection): TrackedDetection? {
        if (previousDetections.isEmpty()) return null

        val currentCenter = getCenterPoint(detection.bbox)
        return previousDetections.minByOrNull { prev ->
            val prevCenter = getCenterPoint(prev.detection.bbox)
            calculateDistance(currentCenter, prevCenter)
        }
    }

    private fun updateTracking(detections: List<Detection>) {
        previousDetections.clear()
        detections.forEach { detection ->
            previousDetections.add(TrackedDetection(detection, frameCounter))
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

    fun getClassLabel(classId: Int): String {
        return if (classId < classLabels.size) classLabels[classId] else "Unknown"
    }

    fun getDetectionCenter(detection: Detection): Pair<Float, Float> {
        return getCenterPoint(detection.bbox)
    }

    fun close() {
        interpreter.close()
        Log.d(TAG, "✅ NEW YOLOv11 detector closed")
    }
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