package com.example.atry.detector

import androidx.compose.ui.graphics.Color
import kotlin.math.*

/**
 * Enhanced utility classes and functions for automatic bar path tracking and analysis
 * Optimized for real-time automatic tracking with intelligent path management
 */

data class PathPoint(
    val x: Float,
    val y: Float,
    val timestamp: Long
) {
    fun distanceTo(other: PathPoint): Float {
        return sqrt((x - other.x).pow(2) + (y - other.y).pow(2))
    }

    fun verticalDistanceTo(other: PathPoint): Float {
        return abs(y - other.y)
    }

    fun horizontalDistanceTo(other: PathPoint): Float {
        return abs(x - other.x)
    }

    fun timeDeltaTo(other: PathPoint): Long {
        return abs(timestamp - other.timestamp)
    }

    fun velocityTo(other: PathPoint): Float {
        val distance = distanceTo(other)
        val timeDelta = timeDeltaTo(other) / 1000f // Convert to seconds
        return if (timeDelta > 0) distance / timeDelta else 0f
    }
}

data class BarPath(
    val id: String = generatePathId(),
    val points: MutableList<PathPoint> = mutableListOf(),
    val isActive: Boolean = true,
    val color: Color = Color.Cyan,
    val startTime: Long = System.currentTimeMillis(),
    val confidence: Float = 1.0f // Track detection confidence for this path
) {
    companion object {
        private var pathCounter = 0
        fun generatePathId(): String = "auto_path_${++pathCounter}"
    }

    fun addPoint(point: PathPoint, maxPoints: Int = 300) {
        points.add(point)

        // Intelligent point management - keep the most recent and important points
        if (points.size > maxPoints) {
            // Keep recent points (last 80%) and some key historical points (first 20%)
            val recentCount = (maxPoints * 0.8).toInt()
            val historicalCount = maxPoints - recentCount

            val recentPoints = points.takeLast(recentCount)
            val historicalPoints = if (historicalCount > 0) {
                // Sample historical points to maintain path shape
                val step = points.size / historicalCount.coerceAtLeast(1)
                points.filterIndexed { index, _ -> index % step == 0 }.take(historicalCount)
            } else emptyList()

            points.clear()
            points.addAll(historicalPoints + recentPoints)
        }
    }

    fun getTotalDistance(): Float {
        if (points.size < 2) return 0f
        return points.zipWithNext { a, b -> a.distanceTo(b) }.sum()
    }

    fun getVerticalRange(): Float {
        if (points.isEmpty()) return 0f
        val minY = points.minOf { it.y }
        val maxY = points.maxOf { it.y }
        return maxY - minY
    }

    fun getHorizontalRange(): Float {
        if (points.isEmpty()) return 0f
        val minX = points.minOf { it.x }
        val maxX = points.maxOf { it.x }
        return maxX - minX
    }

    fun getDuration(): Long {
        if (points.isEmpty()) return 0L
        return points.last().timestamp - points.first().timestamp
    }

    fun getAverageVelocity(): Float {
        if (points.size < 2) return 0f
        val totalDistance = getTotalDistance()
        val duration = getDuration() / 1000f
        return if (duration > 0) totalDistance / duration else 0f
    }

    fun getPeakVelocity(): Float {
        if (points.size < 2) return 0f
        return points.zipWithNext { a, b -> a.velocityTo(b) }.maxOrNull() ?: 0f
    }

    fun getPathDeviation(): Float {
        if (points.isEmpty()) return 0f
        val centerX = points.map { it.x }.average().toFloat()
        return points.map { abs(it.x - centerX) }.average().toFloat()
    }

    fun getPathSmoothness(): Float {
        if (points.size < 3) return 1f

        // Calculate direction changes - smoother paths have fewer sharp direction changes
        var directionChanges = 0
        for (i in 1 until points.size - 1) {
            val prev = points[i - 1]
            val curr = points[i]
            val next = points[i + 1]

            val angle1 = atan2(curr.y - prev.y, curr.x - prev.x)
            val angle2 = atan2(next.y - curr.y, next.x - curr.x)
            val angleDiff = abs(angle1 - angle2)

            if (angleDiff > PI / 4) { // 45 degrees threshold
                directionChanges++
            }
        }

        return 1f - (directionChanges.toFloat() / points.size.coerceAtLeast(1))
    }

    fun isValidForAnalysis(): Boolean {
        return points.size >= 10 &&
                getDuration() > 1000L && // At least 1 second
                getVerticalRange() > 0.02f // Some meaningful movement
    }

    fun getQualityScore(): Float {
        if (!isValidForAnalysis()) return 0f

        val smoothness = getPathSmoothness()
        val consistencyScore = 1f - (getPathDeviation() * 10f).coerceAtMost(1f)
        val durationScore = (getDuration() / 10000f).coerceAtMost(1f) // Up to 10 seconds is good
        val rangeScore = (getVerticalRange() * 5f).coerceAtMost(1f) // Good range of motion

        return (smoothness * 0.3f + consistencyScore * 0.3f + durationScore * 0.2f + rangeScore * 0.2f)
    }
}

enum class MovementDirection {
    UP, DOWN, STABLE
}

data class MovementPhase(
    val direction: MovementDirection,
    val startPoint: PathPoint,
    val endPoint: PathPoint?,
    val maxDisplacement: Float = 0f,
    val duration: Long = 0L,
    val averageVelocity: Float = 0f
)

data class MovementAnalysis(
    val direction: MovementDirection,
    val velocity: Float, // pixels per second
    val acceleration: Float = 0f,
    val totalDistance: Float,
    val repCount: Int,
    val currentPhase: MovementPhase? = null,
    val averageBarSpeed: Float = 0f,
    val peakVelocity: Float = 0f,
    val pathQuality: Float = 0f,
    val confidence: Float = 1f
)

data class LiftingMetrics(
    val totalReps: Int,
    val averageRepTime: Float,
    val averageRangeOfMotion: Float,
    val barPathDeviation: Float,
    val consistencyScore: Float,
    val overallQuality: Float,
    val phases: List<MovementPhase> = emptyList(),
    val sessionDuration: Long = 0L,
    val totalDistance: Float = 0f
)

/**
 * Enhanced bar path analyzer optimized for automatic real-time tracking
 */
class BarPathAnalyzer(
    private val smoothingWindow: Int = 3, // Reduced for faster response
    private val minRepDisplacement: Float = 0.06f, // Slightly reduced threshold
    private val velocityThreshold: Float = 0.01f,
    private val stableThreshold: Float = 0.008f // More sensitive
) {

    private var lastDirection: MovementDirection? = null
    private var repPhases = mutableListOf<MovementPhase>()
    private var currentPhase: MovementPhase? = null
    private var lastAnalysisTime = 0L

    fun analyzeMovement(points: List<PathPoint>): MovementAnalysis {
        val currentTime = System.currentTimeMillis()

        // Throttle analysis for performance (max 10 times per second)
        if (currentTime - lastAnalysisTime < 100L && points.size > 20) {
            // Return cached analysis for very recent calls
            return createBasicAnalysis(points)
        }
        lastAnalysisTime = currentTime

        if (points.size < 5) {
            return createBasicAnalysis(points)
        }

        val smoothedPoints = if (points.size > smoothingWindow * 2) {
            applySmoothingFilter(points)
        } else points

        val direction = calculateDirection(smoothedPoints)
        val velocity = calculateVelocity(smoothedPoints)
        val acceleration = calculateAcceleration(smoothedPoints)
        val totalDistance = calculateTotalDistance(smoothedPoints)
        val repCount = countRepsAdvanced(smoothedPoints)
        val averageSpeed = calculateAverageSpeed(smoothedPoints)
        val peakVelocity = calculatePeakVelocity(smoothedPoints)
        val pathQuality = calculatePathQuality(smoothedPoints)

        return MovementAnalysis(
            direction = direction,
            velocity = velocity,
            acceleration = acceleration,
            totalDistance = totalDistance,
            repCount = repCount,
            currentPhase = currentPhase,
            averageBarSpeed = averageSpeed,
            peakVelocity = peakVelocity,
            pathQuality = pathQuality,
            confidence = calculateConfidence(smoothedPoints)
        )
    }

    private fun createBasicAnalysis(points: List<PathPoint>): MovementAnalysis {
        return MovementAnalysis(
            direction = MovementDirection.STABLE,
            velocity = 0f,
            totalDistance = if (points.size > 1) calculateTotalDistance(points) else 0f,
            repCount = 0,
            pathQuality = 0.5f
        )
    }

    fun analyzeMultiplePaths(paths: List<BarPath>): LiftingMetrics {
        val validPaths = paths.filter { it.isValidForAnalysis() }
        if (validPaths.isEmpty()) {
            return LiftingMetrics(0, 0f, 0f, 0f, 0f, 0f)
        }

        val allPoints = validPaths.flatMap { it.points }
        val totalReps = validPaths.sumOf { countRepsAdvanced(it.points) }
        val averageRepTime = calculateAverageRepTime(validPaths, totalReps)
        val averageROM = validPaths.map { it.getVerticalRange() }.average().toFloat()
        val pathDeviation = validPaths.map { it.getPathDeviation() }.average().toFloat()
        val consistency = calculateConsistencyScore(validPaths)
        val overallQuality = validPaths.map { it.getQualityScore() }.average().toFloat()
        val sessionDuration = if (allPoints.isNotEmpty()) {
            allPoints.maxOf { it.timestamp } - allPoints.minOf { it.timestamp }
        } else 0L
        val totalDistance = validPaths.map { it.getTotalDistance() }.sum()

        return LiftingMetrics(
            totalReps = totalReps,
            averageRepTime = averageRepTime,
            averageRangeOfMotion = averageROM,
            barPathDeviation = pathDeviation,
            consistencyScore = consistency,
            overallQuality = overallQuality,
            phases = repPhases.toList(),
            sessionDuration = sessionDuration,
            totalDistance = totalDistance
        )
    }

    private fun applySmoothingFilter(points: List<PathPoint>): List<PathPoint> {
        if (points.size <= smoothingWindow * 2) return points

        return points.mapIndexed { index, point ->
            val start = maxOf(0, index - smoothingWindow / 2)
            val end = minOf(points.size - 1, index + smoothingWindow / 2)
            val windowPoints = points.subList(start, end + 1)

            val avgX = windowPoints.map { it.x }.average().toFloat()
            val avgY = windowPoints.map { it.y }.average().toFloat()

            PathPoint(avgX, avgY, point.timestamp)
        }
    }

    private fun calculateDirection(points: List<PathPoint>): MovementDirection {
        if (points.size < smoothingWindow) return MovementDirection.STABLE

        val recent = points.takeLast(smoothingWindow.coerceAtMost(points.size))
        if (recent.size < 2) return MovementDirection.STABLE

        val verticalChange = recent.last().y - recent.first().y

        return when {
            verticalChange > stableThreshold -> MovementDirection.DOWN
            verticalChange < -stableThreshold -> MovementDirection.UP
            else -> MovementDirection.STABLE
        }
    }

    private fun calculateVelocity(points: List<PathPoint>): Float {
        if (points.size < 2) return 0f

        val recent = points.takeLast(minOf(8, points.size))
        if (recent.size < 2) return 0f

        val totalDisplacement = recent.zipWithNext { a, b -> a.distanceTo(b) }.sum()
        val timeSpan = (recent.last().timestamp - recent.first().timestamp) / 1000f

        return if (timeSpan > 0) totalDisplacement / timeSpan else 0f
    }

    private fun calculateAcceleration(points: List<PathPoint>): Float {
        if (points.size < 3) return 0f

        val velocities = mutableListOf<Float>()
        for (i in 1 until points.size) {
            val displacement = points[i].distanceTo(points[i-1])
            val timeSpan = (points[i].timestamp - points[i-1].timestamp) / 1000f
            if (timeSpan > 0) {
                velocities.add(displacement / timeSpan)
            }
        }

        if (velocities.size < 2) return 0f

        val recentVelocities = velocities.takeLast(5)
        if (recentVelocities.size < 2) return 0f

        val velocityChange = recentVelocities.last() - recentVelocities.first()
        val timeSpan = recentVelocities.size * 0.1f

        return velocityChange / timeSpan
    }

    private fun calculateTotalDistance(points: List<PathPoint>): Float {
        return points.zipWithNext { a, b -> a.distanceTo(b) }.sum()
    }

    private fun countRepsAdvanced(points: List<PathPoint>): Int {
        if (points.size < 15) return 0

        var repCount = 0
        var inRepPhase = false
        var repStartY: Float? = null
        var currentDirection: MovementDirection? = null

        val smoothed = if (points.size > smoothingWindow * 4) {
            applySmoothingFilter(points)
        } else points

        for (i in smoothingWindow until smoothed.size - smoothingWindow) {
            val prevWindow = smoothed.subList(maxOf(0, i - smoothingWindow), i)
            val nextWindow = smoothed.subList(i, minOf(smoothed.size, i + smoothingWindow))

            if (prevWindow.isNotEmpty() && nextWindow.isNotEmpty()) {
                val prevY = prevWindow.map { it.y }.average()
                val nextY = nextWindow.map { it.y }.average()
                val displacement = nextY - prevY

                val direction = when {
                    displacement > stableThreshold -> MovementDirection.DOWN
                    displacement < -stableThreshold -> MovementDirection.UP
                    else -> MovementDirection.STABLE
                }

                // Enhanced rep detection logic
                when {
                    currentDirection == MovementDirection.DOWN && direction == MovementDirection.UP -> {
                        if (!inRepPhase) {
                            inRepPhase = true
                            repStartY = smoothed[i].y
                        }
                    }
                    currentDirection == MovementDirection.UP && direction == MovementDirection.DOWN -> {
                        if (inRepPhase && repStartY != null) {
                            val totalDisplacement = abs(smoothed[i].y - repStartY!!)
                            if (totalDisplacement > minRepDisplacement) {
                                repCount++

                                val repPhase = MovementPhase(
                                    direction = MovementDirection.UP,
                                    startPoint = PathPoint(0f, repStartY!!, 0L),
                                    endPoint = smoothed[i],
                                    maxDisplacement = totalDisplacement,
                                    duration = if (i > 0) smoothed[i].timestamp - smoothed[maxOf(0, i-10)].timestamp else 0L
                                )
                                repPhases.add(repPhase)
                            }
                            inRepPhase = false
                            repStartY = null
                        }
                    }
                }

                if (direction != MovementDirection.STABLE) {
                    currentDirection = direction
                }
            }
        }

        return repCount
    }

    private fun calculateAverageSpeed(points: List<PathPoint>): Float {
        if (points.size < 2) return 0f

        val totalDistance = calculateTotalDistance(points)
        val totalTime = (points.last().timestamp - points.first().timestamp) / 1000f

        return if (totalTime > 0) totalDistance / totalTime else 0f
    }

    private fun calculatePeakVelocity(points: List<PathPoint>): Float {
        if (points.size < 2) return 0f

        var maxVelocity = 0f

        for (i in 1 until points.size) {
            val displacement = points[i].distanceTo(points[i-1])
            val timeSpan = (points[i].timestamp - points[i-1].timestamp) / 1000f
            if (timeSpan > 0) {
                val velocity = displacement / timeSpan
                maxVelocity = maxOf(maxVelocity, velocity)
            }
        }

        return maxVelocity
    }

    private fun calculatePathQuality(points: List<PathPoint>): Float {
        if (points.size < 5) return 0f

        // Calculate smoothness
        var totalAngleChange = 0f
        var angleCount = 0

        for (i in 1 until points.size - 1) {
            val p1 = points[i - 1]
            val p2 = points[i]
            val p3 = points[i + 1]

            val angle1 = atan2(p2.y - p1.y, p2.x - p1.x)
            val angle2 = atan2(p3.y - p2.y, p3.x - p2.x)
            val angleDiff = abs(angle1 - angle2)

            totalAngleChange += angleDiff
            angleCount++
        }

        val smoothness = if (angleCount > 0) {
            1f - (totalAngleChange / angleCount / PI).toFloat().coerceAtMost(1f)
        } else 1f

        // Calculate consistency (low deviation from center line)
        val centerX = points.map { it.x }.average().toFloat()
        val deviations = points.map { abs(it.x - centerX) }
        val avgDeviation = deviations.average().toFloat()
        val consistency = 1f - (avgDeviation * 10f).coerceAtMost(1f)

        return (smoothness * 0.6f + consistency * 0.4f)
    }

    private fun calculateConfidence(points: List<PathPoint>): Float {
        if (points.size < 3) return 0.5f

        val pathLength = calculateTotalDistance(points)
        val timeSpan = (points.last().timestamp - points.first().timestamp) / 1000f
        val averageVelocity = if (timeSpan > 0) pathLength / timeSpan else 0f

        // Higher confidence for reasonable velocities and sufficient data
        val velocityScore = (averageVelocity / 0.5f).coerceAtMost(1f) // Normalize around 0.5 units/sec
        val dataScore = (points.size / 20f).coerceAtMost(1f) // More points = higher confidence
        val qualityScore = calculatePathQuality(points)

        return (velocityScore * 0.3f + dataScore * 0.3f + qualityScore * 0.4f)
    }

    private fun calculateAverageRepTime(paths: List<BarPath>, repCount: Int): Float {
        if (repCount == 0 || paths.isEmpty()) return 0f

        val totalTime = paths.map { it.getDuration() }.sum() / 1000f
        return totalTime / repCount
    }

    private fun calculateConsistencyScore(paths: List<BarPath>): Float {
        if (paths.size < 2) return 1f

        val pathCharacteristics = paths.map { path ->
            listOf(
                path.getVerticalRange(),
                path.getTotalDistance(),
                path.getPathDeviation(),
                path.getAverageVelocity()
            )
        }

        if (pathCharacteristics.isEmpty()) return 1f

        val consistencyScores = (0..3).map { index ->
            val values = pathCharacteristics.map { it[index] }.filter { it > 0f }
            if (values.isEmpty()) return@map 1f

            val mean = values.average()
            val variance = values.map { (it - mean).pow(2) }.average()
            val stdDev = sqrt(variance)

            val coefficientOfVariation = if (mean > 0) stdDev / mean else 0.0
            maxOf(0f, 1f - coefficientOfVariation.toFloat())
        }

        return consistencyScores.average().toFloat()
    }
}

/**
 * Automatic path session manager for intelligent tracking
 */
class AutomaticPathManager(
    private val maxActivePaths: Int = 3,
    private val pathTimeoutMs: Long = 5000L,
    private val minPathPoints: Int = 10
) {
    private val activePaths = mutableListOf<BarPath>()
    private var lastCleanupTime = 0L

    fun addDetection(detection: Detection, currentTime: Long): List<BarPath> {
        val centerX = (detection.bbox.left + detection.bbox.right) / 2f
        val centerY = (detection.bbox.top + detection.bbox.bottom) / 2f
        val newPoint = PathPoint(centerX, centerY, currentTime)

        // Find closest active path or create new one
        val targetPath = findClosestPath(newPoint) ?: createNewPath(currentTime)

        // Add point to path
        targetPath.addPoint(newPoint)

        // Ensure path is in active list
        if (!activePaths.contains(targetPath)) {
            activePaths.add(targetPath)
        }

        // Cleanup old paths periodically
        if (currentTime - lastCleanupTime > 2000L) {
            cleanupOldPaths(currentTime)
            lastCleanupTime = currentTime
        }

        return activePaths.toList()
    }

    private fun findClosestPath(newPoint: PathPoint): BarPath? {
        if (activePaths.isEmpty()) return null

        val recentPaths = activePaths.filter { path ->
            path.points.isNotEmpty() &&
                    newPoint.timestamp - path.points.last().timestamp < 2000L // Within 2 seconds
        }

        return recentPaths.minByOrNull { path ->
            if (path.points.isNotEmpty()) {
                newPoint.distanceTo(path.points.last())
            } else Float.MAX_VALUE
        }?.takeIf { path ->
            if (path.points.isNotEmpty()) {
                newPoint.distanceTo(path.points.last()) < 0.1f // Within reasonable distance
            } else false
        }
    }

    private fun createNewPath(currentTime: Long): BarPath {
        // Limit number of active paths
        if (activePaths.size >= maxActivePaths) {
            // Remove oldest path that hasn't been updated recently
            val oldestPath = activePaths.minByOrNull { path ->
                if (path.points.isNotEmpty()) path.points.last().timestamp else 0L
            }
            oldestPath?.let { activePaths.remove(it) }
        }

        val newPath = BarPath(
            color = getColorForPathIndex(activePaths.size),
            startTime = currentTime,
            confidence = 1.0f
        )

        return newPath
    }

    private fun cleanupOldPaths(currentTime: Long) {
        // Remove paths that haven't been updated recently or are too short
        activePaths.removeAll { path ->
            val isOld = path.points.isEmpty() ||
                    currentTime - path.points.last().timestamp > pathTimeoutMs
            val isTooShort = path.points.size < minPathPoints &&
                    currentTime - path.startTime > 3000L // Give 3 seconds minimum

            isOld || isTooShort
        }

        // Trim points from remaining paths to manage memory
        activePaths.forEach { path ->
            if (path.points.size > 500) {
                val keepPoints = path.points.takeLast(300)
                path.points.clear()
                path.points.addAll(keepPoints)
            }
        }
    }

    private fun getColorForPathIndex(index: Int): Color {
        val colors = listOf(
            Color.Cyan,
            Color.Yellow,
            Color.Green,
            Color.Magenta,
            Color.Red,
            Color.Blue,
            Color.White
        )
        return colors[index % colors.size]
    }

    fun getCurrentPaths(): List<BarPath> = activePaths.toList()

    fun clearAllPaths() {
        activePaths.clear()
    }

    fun getActivePathCount(): Int = activePaths.size

    fun getTotalPoints(): Int = activePaths.map { it.points.size }.sum()

    fun getSessionStatistics(): SessionStatistics {
        val allPoints = activePaths.flatMap { it.points }
        val totalReps = activePaths.map { path ->
            if (path.points.size > 20) {
                val analyzer = BarPathAnalyzer()
                analyzer.analyzeMovement(path.points).repCount
            } else 0
        }.sum()

        val sessionDuration = if (allPoints.isNotEmpty()) {
            allPoints.maxOf { it.timestamp } - allPoints.minOf { it.timestamp }
        } else 0L

        val totalDistance = activePaths.map { it.getTotalDistance() }.sum()
        val averageQuality = if (activePaths.isNotEmpty()) {
            activePaths.map { it.getQualityScore() }.average().toFloat()
        } else 0f

        return SessionStatistics(
            totalPaths = activePaths.size,
            totalPoints = allPoints.size,
            totalReps = totalReps,
            sessionDuration = sessionDuration,
            totalDistance = totalDistance,
            averageQuality = averageQuality,
            activeTracking = activePaths.any { path ->
                path.points.isNotEmpty() &&
                        System.currentTimeMillis() - path.points.last().timestamp < 3000L
            }
        )
    }
}

/**
 * Session statistics data class
 */
data class SessionStatistics(
    val totalPaths: Int,
    val totalPoints: Int,
    val totalReps: Int,
    val sessionDuration: Long,
    val totalDistance: Float,
    val averageQuality: Float,
    val activeTracking: Boolean
)

/**
 * Utility functions for automatic tracking
 */
object AutoTrackingUtils {

    /**
     * Determines if a detection is likely to be part of a barbell movement
     */
    fun isValidBarbellDetection(
        detection: Detection,
        previousDetections: List<Detection>,
        timeWindowMs: Long = 1000L
    ): Boolean {
        // Check detection confidence
        if (detection.score < 0.3f) return false

        // Check bounding box dimensions (barbells should have reasonable aspect ratios)
        val bbox = detection.bbox
        val width = bbox.right - bbox.left
        val height = bbox.bottom - bbox.top
        val aspectRatio = width / height

        // Barbells typically have width > height, but allow some flexibility
        if (aspectRatio < 0.5f || aspectRatio > 10f) return false

        // Check if detection is reasonably sized
        val area = width * height
        if (area < 0.001f || area > 0.5f) return false

        // Check consistency with recent detections
        if (previousDetections.isNotEmpty()) {
            val currentCenter = Pair(
                (bbox.left + bbox.right) / 2f,
                (bbox.top + bbox.bottom) / 2f
            )

            // Find most recent similar detection
            val recentSimilar = previousDetections.lastOrNull { prev ->
                val prevCenter = Pair(
                    (prev.bbox.left + prev.bbox.right) / 2f,
                    (prev.bbox.top + prev.bbox.bottom) / 2f
                )
                val distance = sqrt(
                    (currentCenter.first - prevCenter.first).pow(2) +
                            (currentCenter.second - prevCenter.second).pow(2)
                )
                distance < 0.2f // Must be within reasonable distance
            }

            // Require some consistency
            if (recentSimilar == null) return false
        }

        return true
    }

    /**
     * Calculates movement smoothness for a path
     */
    fun calculateMovementSmoothness(points: List<PathPoint>): Float {
        if (points.size < 3) return 0f

        var totalCurvature = 0f
        var curvatureCount = 0

        for (i in 1 until points.size - 1) {
            val p1 = points[i - 1]
            val p2 = points[i]
            val p3 = points[i + 1]

            // Calculate curvature approximation
            val v1x = p2.x - p1.x
            val v1y = p2.y - p1.y
            val v2x = p3.x - p2.x
            val v2y = p3.y - p2.y

            val crossProduct = abs(v1x * v2y - v1y * v2x)
            val dotProduct = v1x * v2x + v1y * v2y

            val curvature = if (dotProduct != 0f) crossProduct / (dotProduct + 1f) else 0f
            totalCurvature += curvature
            curvatureCount++
        }

        val averageCurvature = if (curvatureCount > 0) totalCurvature / curvatureCount else 0f
        return (1f - averageCurvature.coerceAtMost(1f))
    }

    /**
     * Predicts next position based on movement pattern
     */
    fun predictNextPosition(
        points: List<PathPoint>,
        timeAheadMs: Long = 100L
    ): Pair<Float, Float>? {
        if (points.size < 3) return null

        val recent = points.takeLast(5)
        if (recent.size < 2) return null

        // Calculate average velocity
        val velocities = recent.zipWithNext { a, b ->
            val dt = (b.timestamp - a.timestamp) / 1000f
            if (dt > 0) {
                Pair((b.x - a.x) / dt, (b.y - a.y) / dt)
            } else {
                Pair(0f, 0f)
            }
        }

        val avgVelX = velocities.map { it.first }.average().toFloat()
        val avgVelY = velocities.map { it.second }.average().toFloat()

        val lastPoint = recent.last()
        val timeAheadSec = timeAheadMs / 1000f

        return Pair(
            lastPoint.x + avgVelX * timeAheadSec,
            lastPoint.y + avgVelY * timeAheadSec
        )
    }

    /**
     * Detects if the barbell movement has stopped
     */
    fun isMovementStopped(
        points: List<PathPoint>,
        timeWindowMs: Long = 2000L,
        movementThreshold: Float = 0.01f
    ): Boolean {
        if (points.isEmpty()) return true

        val currentTime = System.currentTimeMillis()
        val recentPoints = points.filter {
            currentTime - it.timestamp <= timeWindowMs
        }

        if (recentPoints.size < 3) return true

        // Check if recent movement is below threshold
        val totalMovement = recentPoints.zipWithNext { a, b -> a.distanceTo(b) }.sum()
        val averageMovement = totalMovement / recentPoints.size

        return averageMovement < movementThreshold
    }
}