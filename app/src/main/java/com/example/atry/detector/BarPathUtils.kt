package com.example.atry.detector

import androidx.compose.ui.graphics.Color
import kotlin.math.*

/**
 * Utility classes and functions for bar path tracking and analysis
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
}

data class BarPath(
    val id: String = generatePathId(),
    val points: MutableList<PathPoint> = mutableListOf(),
    val isActive: Boolean = true,
    val color: Color = Color.Cyan,
    val startTime: Long = System.currentTimeMillis()
) {
    companion object {
        private var pathCounter = 0
        fun generatePathId(): String = "path_${++pathCounter}"
    }

    fun addPoint(point: PathPoint, maxPoints: Int = 500) {
        points.add(point)
        if (points.size > maxPoints) {
            points.removeAt(0)
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

    fun getDuration(): Long {
        if (points.isEmpty()) return 0L
        return points.last().timestamp - points.first().timestamp
    }
}

enum class MovementDirection {
    UP, DOWN, STABLE
}

data class MovementPhase(
    val direction: MovementDirection,
    val startPoint: PathPoint,
    val endPoint: PathPoint?,
    val maxDisplacement: Float = 0f
)

data class MovementAnalysis(
    val direction: MovementDirection,
    val velocity: Float, // pixels per second
    val acceleration: Float = 0f,
    val totalDistance: Float,
    val repCount: Int,
    val currentPhase: MovementPhase? = null,
    val averageBarSpeed: Float = 0f,
    val peakVelocity: Float = 0f
)

data class LiftingMetrics(
    val totalReps: Int,
    val averageRepTime: Float,
    val averageRangeOfMotion: Float,
    val barPathDeviation: Float, // How much the bar deviates from vertical
    val consistencyScore: Float, // How consistent the movement pattern is
    val phases: List<MovementPhase> = emptyList()
)

/**
 * Advanced bar path analyzer with comprehensive movement analysis
 */
class BarPathAnalyzer(
    private val smoothingWindow: Int = 5,
    private val minRepDisplacement: Float = 0.08f, // Minimum vertical displacement for a rep
    private val velocityThreshold: Float = 0.01f,
    private val stableThreshold: Float = 0.005f
) {

    private var lastDirection: MovementDirection? = null
    private var repPhases = mutableListOf<MovementPhase>()
    private var currentPhase: MovementPhase? = null

    fun analyzeMovement(points: List<PathPoint>): MovementAnalysis {
        if (points.size < 3) {
            return MovementAnalysis(
                direction = MovementDirection.STABLE,
                velocity = 0f,
                totalDistance = 0f,
                repCount = 0
            )
        }

        val smoothedPoints = applySmoothingFilter(points)
        val direction = calculateDirection(smoothedPoints)
        val velocity = calculateVelocity(smoothedPoints)
        val acceleration = calculateAcceleration(smoothedPoints)
        val totalDistance = calculateTotalDistance(smoothedPoints)
        val repCount = countRepsAdvanced(smoothedPoints)
        val averageSpeed = calculateAverageSpeed(smoothedPoints)
        val peakVelocity = calculatePeakVelocity(smoothedPoints)

        return MovementAnalysis(
            direction = direction,
            velocity = velocity,
            acceleration = acceleration,
            totalDistance = totalDistance,
            repCount = repCount,
            currentPhase = currentPhase,
            averageBarSpeed = averageSpeed,
            peakVelocity = peakVelocity
        )
    }

    fun calculateLiftingMetrics(paths: List<BarPath>): LiftingMetrics {
        val allPoints = paths.flatMap { it.points }
        if (allPoints.isEmpty()) {
            return LiftingMetrics(0, 0f, 0f, 0f, 0f)
        }

        val totalReps = countRepsAdvanced(allPoints)
        val averageRepTime = calculateAverageRepTime(allPoints, totalReps)
        val averageROM = calculateAverageRangeOfMotion(paths)
        val pathDeviation = calculateBarPathDeviation(allPoints)
        val consistency = calculateConsistencyScore(paths)

        return LiftingMetrics(
            totalReps = totalReps,
            averageRepTime = averageRepTime,
            averageRangeOfMotion = averageROM,
            barPathDeviation = pathDeviation,
            consistencyScore = consistency,
            phases = repPhases.toList()
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

        val recent = points.takeLast(smoothingWindow)
        val verticalChange = recent.last().y - recent.first().y

        return when {
            verticalChange > stableThreshold -> MovementDirection.DOWN
            verticalChange < -stableThreshold -> MovementDirection.UP
            else -> MovementDirection.STABLE
        }
    }

    private fun calculateVelocity(points: List<PathPoint>): Float {
        if (points.size < 2) return 0f

        val recent = points.takeLast(minOf(10, points.size))
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
        val velocityChange = recentVelocities.last() - recentVelocities.first()
        val timeSpan = recentVelocities.size * 0.1f // Assuming ~10 FPS

        return velocityChange / timeSpan
    }

    private fun calculateTotalDistance(points: List<PathPoint>): Float {
        return points.zipWithNext { a, b -> a.distanceTo(b) }.sum()
    }

    private fun countRepsAdvanced(points: List<PathPoint>): Int {
        if (points.size < 20) return 0

        var repCount = 0
        var inRepPhase = false
        var repStartY: Float? = null
        var currentDirection: MovementDirection? = null

        val smoothed = applySmoothingFilter(points)

        for (i in smoothingWindow until smoothed.size - smoothingWindow) {
            // FIXED: Convert smoothingWindow to Int and ensure proper indexing
            val prevY = smoothed.subList(i - smoothingWindow, i).map { it.y }.average()
            val nextY = smoothed.subList(i, i + smoothingWindow).map { it.y }.average()
            val displacement = nextY - prevY

            val direction = when {
                displacement > stableThreshold -> MovementDirection.DOWN
                displacement < -stableThreshold -> MovementDirection.UP
                else -> MovementDirection.STABLE
            }

            // Detect phase transitions
            if (currentDirection != direction && direction != MovementDirection.STABLE) {
                when {
                    // Starting upward phase (concentric)
                    currentDirection == MovementDirection.DOWN && direction == MovementDirection.UP -> {
                        if (!inRepPhase) {
                            inRepPhase = true
                            repStartY = smoothed[i].y
                        }
                    }
                    // Completing downward phase (eccentric) - rep completed
                    currentDirection == MovementDirection.UP && direction == MovementDirection.DOWN -> {
                        if (inRepPhase && repStartY != null) {
                            val totalDisplacement = abs(smoothed[i].y - repStartY!!)
                            if (totalDisplacement > minRepDisplacement) {
                                repCount++

                                // Record the rep phase
                                val repPhase = MovementPhase(
                                    direction = MovementDirection.UP,
                                    startPoint = PathPoint(0f, repStartY!!, 0L),
                                    endPoint = smoothed[i],
                                    maxDisplacement = totalDisplacement
                                )
                                repPhases.add(repPhase)
                            }
                            inRepPhase = false
                            repStartY = null
                        }
                    }
                }
                currentDirection = direction
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

    private fun calculateAverageRepTime(points: List<PathPoint>, repCount: Int): Float {
        if (repCount == 0 || points.isEmpty()) return 0f

        val totalTime = (points.last().timestamp - points.first().timestamp) / 1000f
        return totalTime / repCount
    }

    private fun calculateAverageRangeOfMotion(paths: List<BarPath>): Float {
        if (paths.isEmpty()) return 0f

        val ranges = paths.map { it.getVerticalRange() }.filter { it > 0f }
        return if (ranges.isNotEmpty()) ranges.average().toFloat() else 0f
    }

    private fun calculateBarPathDeviation(points: List<PathPoint>): Float {
        if (points.isEmpty()) return 0f

        // Calculate how much the bar deviates from a straight vertical line
        val centerX = points.map { it.x }.average().toFloat()
        val deviations = points.map { abs(it.x - centerX) }

        return deviations.average().toFloat()
    }

    private fun calculateConsistencyScore(paths: List<BarPath>): Float {
        if (paths.size < 2) return 1f

        // Calculate consistency based on path similarity
        val pathCharacteristics = paths.map { path ->
            listOf(
                path.getVerticalRange(),
                path.getTotalDistance(),
                calculateBarPathDeviation(path.points)
            )
        }

        if (pathCharacteristics.isEmpty()) return 1f

        // Calculate coefficient of variation for each characteristic
        val consistencyScores = (0..2).map { index ->
            val values = pathCharacteristics.map { it[index] }.filter { it > 0f }
            if (values.isEmpty()) return@map 1f

            val mean = values.average()
            val variance = values.map { (it - mean).pow(2) }.average()
            val stdDev = sqrt(variance)

            // Lower coefficient of variation = higher consistency
            val coefficientOfVariation = if (mean > 0) stdDev / mean else 0.0
            maxOf(0f, 1f - coefficientOfVariation.toFloat())
        }

        return consistencyScores.average().toFloat()
    }
}