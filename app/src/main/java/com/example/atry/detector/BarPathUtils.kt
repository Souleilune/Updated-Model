package com.example.atry.detector

import androidx.compose.ui.graphics.Color
import kotlin.math.*

/**
 * Simplified Rep Counter based on vertical position changes
 * Point A (bottom) -> Point B (top) -> Point A (bottom) = 1 Rep
 * Much more practical and achievable for real lifting scenarios
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

    fun addPoint(point: PathPoint, maxPoints: Int = 300) {
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

enum class RepPhase {
    AT_BOTTOM,    // Starting position (low Y value)
    MOVING_UP,    // Moving from bottom to top
    AT_TOP,       // Peak position (high Y value)
    MOVING_DOWN,  // Moving from top back to bottom
    COMPLETED     // Full rep completed
}

data class RepState(
    var phase: RepPhase = RepPhase.AT_BOTTOM,
    var startY: Float = 0f,
    var peakY: Float = 0f,
    var currentY: Float = 0f,
    var lastDirection: MovementDirection = MovementDirection.STABLE,
    var phaseStartTime: Long = 0L
)

data class MovementAnalysis(
    val direction: MovementDirection,
    val velocity: Float,
    val acceleration: Float = 0f,
    val totalDistance: Float,
    val repCount: Int,
    val currentPhase: RepPhase = RepPhase.AT_BOTTOM,
    val averageBarSpeed: Float = 0f,
    val peakVelocity: Float = 0f
)

/**
 * Simplified Bar Path Analyzer focused on practical rep counting
 * Much faster and more reliable than complex movement analysis
 */
class SimpleRepAnalyzer(
    private val minVerticalMovement: Float = 0.08f,  // Minimum movement to count as rep (8% of screen)
    private val stableThreshold: Float = 0.02f,      // Movement threshold for direction changes
    private val smoothingWindow: Int = 3             // Small window for smoothing
) {

    private var repState = RepState()
    private var totalReps = 0

    companion object {
        private const val TAG = "SimpleRepAnalyzer"
    }

    /**
     * Simple and fast movement analysis
     */
    fun analyzeMovement(points: List<PathPoint>): MovementAnalysis? {
        if (points.size < 5) return null

        val direction = calculateSimpleDirection(points)
        val velocity = calculateSimpleVelocity(points)
        val totalDistance = calculateTotalDistance(points)
        val repCount = countSimpleReps(points)

        return MovementAnalysis(
            direction = direction,
            velocity = velocity,
            totalDistance = totalDistance,
            repCount = repCount,
            currentPhase = repState.phase
        )
    }

    /**
     * SIMPLE REP COUNTING: Bottom -> Top -> Bottom = 1 Rep
     */
    private fun countSimpleReps(points: List<PathPoint>): Int {
        if (points.size < 10) return totalReps

        val currentPoint = points.last()
        val currentY = currentPoint.y
        val currentTime = currentPoint.timestamp

        // Get recent movement direction
        val recentDirection = calculateSimpleDirection(points.takeLast(smoothingWindow))

        // Update rep state based on vertical position and direction
        when (repState.phase) {
            RepPhase.AT_BOTTOM -> {
                // Starting at bottom, wait for upward movement
                if (recentDirection == MovementDirection.UP) {
                    repState.phase = RepPhase.MOVING_UP
                    repState.startY = currentY
                    repState.phaseStartTime = currentTime
                    android.util.Log.d(TAG, "Rep started: Moving up from Y=${String.format("%.3f", currentY)}")
                }
            }

            RepPhase.MOVING_UP -> {
                // Moving up, look for peak or direction change
                if (recentDirection == MovementDirection.DOWN) {
                    // Reached peak, now moving down
                    if (currentY - repState.startY >= minVerticalMovement) {
                        repState.phase = RepPhase.MOVING_DOWN
                        repState.peakY = currentY
                        android.util.Log.d(TAG, "Reached peak: Y=${String.format("%.3f", currentY)}, Range=${String.format("%.3f", currentY - repState.startY)}")
                    } else {
                        // Movement too small, reset
                        repState.phase = RepPhase.AT_BOTTOM
                        android.util.Log.d(TAG, "Movement too small, resetting")
                    }
                }
            }

            RepPhase.MOVING_DOWN -> {
                // Moving down, check if we've returned close to starting position
                val returnedToStart = currentY <= repState.startY + (minVerticalMovement * 0.3f)
                val hasMinimumRange = repState.peakY - repState.startY >= minVerticalMovement

                if (returnedToStart && hasMinimumRange) {
                    // COMPLETED ONE REP!
                    totalReps++
                    repState.phase = RepPhase.AT_BOTTOM
                    val repTime = (currentTime - repState.phaseStartTime) / 1000f
                    android.util.Log.d(TAG, "🎯 REP COMPLETED! Total: $totalReps, Time: ${String.format("%.1f", repTime)}s, Range: ${String.format("%.3f", repState.peakY - repState.startY)}")

                    // Reset for next rep
                    repState.startY = currentY
                }
            }

            else -> {
                repState.phase = RepPhase.AT_BOTTOM
            }
        }

        repState.currentY = currentY
        repState.lastDirection = recentDirection

        return totalReps
    }

    /**
     * Simple direction calculation based on recent Y movement
     */
    private fun calculateSimpleDirection(points: List<PathPoint>): MovementDirection {
        if (points.size < 2) return MovementDirection.STABLE

        val recent = points.takeLast(minOf(smoothingWindow, points.size))
        if (recent.size < 2) return MovementDirection.STABLE

        val startY = recent.first().y
        val endY = recent.last().y
        val verticalChange = endY - startY

        return when {
            verticalChange > stableThreshold -> MovementDirection.DOWN
            verticalChange < -stableThreshold -> MovementDirection.UP
            else -> MovementDirection.STABLE
        }
    }

    /**
     * Simple velocity calculation
     */
    private fun calculateSimpleVelocity(points: List<PathPoint>): Float {
        if (points.size < 2) return 0f

        val recent = points.takeLast(minOf(5, points.size))
        if (recent.size < 2) return 0f

        val distance = recent.first().distanceTo(recent.last())
        val timeSpan = (recent.last().timestamp - recent.first().timestamp) / 1000f

        return if (timeSpan > 0) distance / timeSpan else 0f
    }

    private fun calculateTotalDistance(points: List<PathPoint>): Float {
        return points.zipWithNext { a, b -> a.distanceTo(b) }.sum()
    }

    /**
     * Reset rep counter (useful for new sessions)
     */
    fun resetRepCounter() {
        totalReps = 0
        repState = RepState()
        android.util.Log.d(TAG, "Rep counter reset")
    }

    /**
     * Get current rep progress info
     */
    fun getRepProgress(): RepProgress {
        return RepProgress(
            currentPhase = repState.phase,
            verticalProgress = if (repState.phase != RepPhase.AT_BOTTOM) {
                abs(repState.currentY - repState.startY) / minVerticalMovement
            } else 0f,
            isInRep = repState.phase != RepPhase.AT_BOTTOM
        )
    }
}

data class RepProgress(
    val currentPhase: RepPhase,
    val verticalProgress: Float,  // 0.0 to 1.0+ (how much of minimum movement completed)
    val isInRep: Boolean
)

/**
 * Enhanced Movement Analysis with simple rep counting
 */
data class LiftingMetrics(
    val totalReps: Int,
    val averageRepTime: Float,
    val averageRangeOfMotion: Float,
    val barPathDeviation: Float,
    val consistencyScore: Float
)

/**
 * Lightweight bar path analyzer for mobile performance
 */
class BarPathAnalyzer {
    private val repAnalyzer = SimpleRepAnalyzer()

    fun analyzeMovement(points: List<PathPoint>): MovementAnalysis? {
        return repAnalyzer.analyzeMovement(points)
    }

    fun resetSession() {
        repAnalyzer.resetRepCounter()
    }

    fun getRepProgress(): RepProgress {
        return repAnalyzer.getRepProgress()
    }

    fun calculateLiftingMetrics(paths: List<BarPath>): LiftingMetrics {
        if (paths.isEmpty()) {
            return LiftingMetrics(0, 0f, 0f, 0f, 0f)
        }

        val totalReps = paths.map { path ->
            if (path.points.size > 10) {
                analyzeMovement(path.points)?.repCount ?: 0
            } else 0
        }.sum()

        val averageRepTime = if (totalReps > 0) {
            val totalTime = paths.map { it.getDuration() }.sum() / 1000f
            totalTime / totalReps
        } else 0f

        val averageROM = paths.map { it.getVerticalRange() }.average().toFloat()

        val barPathDeviation = paths.map { path ->
            if (path.points.isEmpty()) 0f else {
                val centerX = path.points.map { it.x }.average().toFloat()
                path.points.map { abs(it.x - centerX) }.average().toFloat()
            }
        }.average().toFloat()

        val consistencyScore = if (paths.size > 1) {
            val ranges = paths.map { it.getVerticalRange() }
            val mean = ranges.average()
            val variance = ranges.map { (it - mean) * (it - mean) }.average()
            val stdDev = sqrt(variance)
            maxOf(0f, 1f - (stdDev / mean).toFloat())
        } else 1f

        return LiftingMetrics(
            totalReps = totalReps,
            averageRepTime = averageRepTime,
            averageRangeOfMotion = averageROM,
            barPathDeviation = barPathDeviation,
            consistencyScore = consistencyScore
        )
    }
}