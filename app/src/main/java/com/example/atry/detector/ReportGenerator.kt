package com.example.atry.detector

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Environment
import android.util.Log
import androidx.core.content.FileProvider
import org.apache.poi.ss.usermodel.*
import org.apache.poi.xssf.usermodel.XSSFWorkbook
import org.apache.poi.xssf.usermodel.XSSFCellStyle
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

/**
 * Android-compatible report generator for bar path analysis
 * Supports both Excel (.xlsx) and CSV formats
 * FIXED: Removed autoSizeColumn() calls that cause crashes on Android
 */
class ReportGenerator(private val context: Context) {

    companion object {
        private const val TAG = "ReportGenerator"
        private const val PROVIDER_AUTHORITY = "com.example.atry.fileprovider"
    }

    data class WorkoutSession(
        val startTime: Long,
        val endTime: Long,
        val totalReps: Int,
        val paths: List<BarPath>,
        val movements: List<MovementAnalysis>,
        val sessionNotes: String = ""
    )

    /**
     * Generate comprehensive Excel report (Android-compatible)
     */
    fun generateExcelReport(
        session: WorkoutSession,
        analyzer: BarPathAnalyzer
    ): Result<File> {
        return try {
            val workbook = XSSFWorkbook()

            // Create multiple sheets for comprehensive analysis
            createSummarySheet(workbook, session, analyzer)
            createDetailedPathSheet(workbook, session)
            createMovementAnalysisSheet(workbook, session)
            createRepAnalysisSheet(workbook, session, analyzer)
            createStatisticsSheet(workbook, session, analyzer)

            // Save to file
            val file = saveWorkbookToFile(workbook, "barpath_report")
            workbook.close()

            Log.d(TAG, "Excel report generated successfully: ${file.absolutePath}")
            Result.success(file)
        } catch (e: Exception) {
            Log.e(TAG, "Error generating Excel report: ${e.message}", e)
            Result.failure(e)
        }
    }

    /**
     * Generate CSV report (simpler format)
     */
    fun generateCSVReport(
        session: WorkoutSession,
        analyzer: BarPathAnalyzer
    ): Result<File> {
        return try {
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val fileName = "barpath_report_$timestamp.csv"
            val file = File(context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), fileName)

            val csv = StringBuilder()

            // Header information
            csv.appendLine("Bar Path Analysis Report")
            csv.appendLine("Generated: ${SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())}")
            csv.appendLine("Session Duration: ${formatDuration(session.endTime - session.startTime)}")
            csv.appendLine("Total Reps: ${session.totalReps}")
            csv.appendLine("")

            // Lifting metrics
            val metrics = analyzer.calculateLiftingMetrics(session.paths)
            csv.appendLine("LIFTING METRICS")
            csv.appendLine("Metric,Value")
            csv.appendLine("Total Reps,${metrics.totalReps}")
            csv.appendLine("Average Rep Time (s),${String.format("%.2f", metrics.averageRepTime)}")
            csv.appendLine("Average Range of Motion,${String.format("%.3f", metrics.averageRangeOfMotion)}")
            csv.appendLine("Bar Path Deviation,${String.format("%.3f", metrics.barPathDeviation)}")
            csv.appendLine("Consistency Score,${String.format("%.3f", metrics.consistencyScore)}")
            csv.appendLine("")

            // Detailed path data
            csv.appendLine("DETAILED PATH DATA")
            csv.appendLine("Rep,Point_Index,X_Position,Y_Position,Timestamp,Distance_From_Previous")

            session.paths.forEachIndexed { pathIndex, path ->
                path.points.forEachIndexed { pointIndex, point ->
                    val distanceFromPrevious = if (pointIndex > 0) {
                        point.distanceTo(path.points[pointIndex - 1])
                    } else 0f

                    csv.appendLine("${pathIndex + 1},$pointIndex,${point.x},${point.y},${point.timestamp},$distanceFromPrevious")
                }
            }

            file.writeText(csv.toString())
            Log.d(TAG, "CSV report generated successfully: ${file.absolutePath}")
            Result.success(file)
        } catch (e: Exception) {
            Log.e(TAG, "Error generating CSV report: ${e.message}", e)
            Result.failure(e)
        }
    }

    private fun createSummarySheet(
        workbook: XSSFWorkbook,
        session: WorkoutSession,
        analyzer: BarPathAnalyzer
    ) {
        val sheet = workbook.createSheet("Summary")
        val headerStyle = createHeaderStyle(workbook)
        val dataStyle = createDataStyle(workbook)

        var rowNum = 0

        // Title
        val titleRow = sheet.createRow(rowNum++)
        val titleCell = titleRow.createCell(0)
        titleCell.setCellValue("Bar Path Analysis Report")
        titleCell.cellStyle = createTitleStyle(workbook)
        sheet.addMergedRegion(org.apache.poi.ss.util.CellRangeAddress(0, 0, 0, 3))

        rowNum++ // Empty row

        // Session info
        val sessionInfoData = arrayOf(
            arrayOf("Generated", SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())),
            arrayOf("Session Start", SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(session.startTime))),
            arrayOf("Session End", SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(session.endTime))),
            arrayOf("Duration", formatDuration(session.endTime - session.startTime)),
            arrayOf("Total Reps", session.totalReps.toString())
        )

        // Session info header
        val sessionHeaderRow = sheet.createRow(rowNum++)
        val sessionHeaderCell = sessionHeaderRow.createCell(0)
        sessionHeaderCell.setCellValue("SESSION INFORMATION")
        sessionHeaderCell.cellStyle = headerStyle

        sessionInfoData.forEach { data ->
            val row = sheet.createRow(rowNum++)
            row.createCell(0).apply { setCellValue(data[0]); cellStyle = dataStyle }
            row.createCell(1).apply { setCellValue(data[1]); cellStyle = dataStyle }
        }

        rowNum++ // Empty row

        // Lifting metrics
        val metrics = analyzer.calculateLiftingMetrics(session.paths)
        val metricsHeaderRow = sheet.createRow(rowNum++)
        val metricsHeaderCell = metricsHeaderRow.createCell(0)
        metricsHeaderCell.setCellValue("LIFTING METRICS")
        metricsHeaderCell.cellStyle = headerStyle

        val metricsData = arrayOf(
            arrayOf("Total Reps", metrics.totalReps.toString()),
            arrayOf("Average Rep Time (s)", String.format("%.2f", metrics.averageRepTime)),
            arrayOf("Average Range of Motion", String.format("%.3f", metrics.averageRangeOfMotion)),
            arrayOf("Bar Path Deviation", String.format("%.3f", metrics.barPathDeviation)),
            arrayOf("Consistency Score", String.format("%.3f", metrics.consistencyScore))
        )

        metricsData.forEach { data ->
            val row = sheet.createRow(rowNum++)
            row.createCell(0).apply { setCellValue(data[0]); cellStyle = dataStyle }
            row.createCell(1).apply { setCellValue(data[1]); cellStyle = dataStyle }
        }

        // REMOVED: Auto-size columns (causes Android crashes)
        // Manual column sizing instead
        sheet.setColumnWidth(0, 6000) // Column A
        sheet.setColumnWidth(1, 4000) // Column B
        sheet.setColumnWidth(2, 3000) // Column C
        sheet.setColumnWidth(3, 3000) // Column D
    }

    private fun createDetailedPathSheet(workbook: XSSFWorkbook, session: WorkoutSession) {
        val sheet = workbook.createSheet("Path Data")
        val headerStyle = createHeaderStyle(workbook)
        val dataStyle = createDataStyle(workbook)

        var rowNum = 0

        // Headers
        val headerRow = sheet.createRow(rowNum++)
        val headers = arrayOf("Rep", "Point_Index", "X_Position", "Y_Position", "Timestamp", "Distance_From_Previous", "Time_Delta")
        headers.forEachIndexed { index, header ->
            val cell = headerRow.createCell(index)
            cell.setCellValue(header)
            cell.cellStyle = headerStyle
        }

        // Data
        session.paths.forEachIndexed { pathIndex, path ->
            path.points.forEachIndexed { pointIndex, point ->
                val row = sheet.createRow(rowNum++)

                val distanceFromPrevious = if (pointIndex > 0) {
                    point.distanceTo(path.points[pointIndex - 1])
                } else 0f

                val timeDelta = if (pointIndex > 0) {
                    point.timestamp - path.points[pointIndex - 1].timestamp
                } else 0L

                row.createCell(0).apply { setCellValue((pathIndex + 1).toDouble()); cellStyle = dataStyle }
                row.createCell(1).apply { setCellValue(pointIndex.toDouble()); cellStyle = dataStyle }
                row.createCell(2).apply { setCellValue(point.x.toDouble()); cellStyle = dataStyle }
                row.createCell(3).apply { setCellValue(point.y.toDouble()); cellStyle = dataStyle }
                row.createCell(4).apply { setCellValue(point.timestamp.toDouble()); cellStyle = dataStyle }
                row.createCell(5).apply { setCellValue(distanceFromPrevious.toDouble()); cellStyle = dataStyle }
                row.createCell(6).apply { setCellValue(timeDelta.toDouble()); cellStyle = dataStyle }
            }
        }

        // Manual column sizing (Android-compatible)
        headers.forEachIndexed { index, _ ->
            sheet.setColumnWidth(index, 3000)
        }
    }

    private fun createMovementAnalysisSheet(workbook: XSSFWorkbook, session: WorkoutSession) {
        val sheet = workbook.createSheet("Movement Analysis")
        val headerStyle = createHeaderStyle(workbook)
        val dataStyle = createDataStyle(workbook)

        var rowNum = 0

        // Headers
        val headerRow = sheet.createRow(rowNum++)
        val headers = arrayOf("Rep", "Direction", "Velocity", "Acceleration", "Total_Distance", "Peak_Velocity", "Average_Speed")
        headers.forEachIndexed { index, header ->
            val cell = headerRow.createCell(index)
            cell.setCellValue(header)
            cell.cellStyle = headerStyle
        }

        // Analyze each path
        val analyzer = BarPathAnalyzer()
        session.paths.forEachIndexed { pathIndex, path ->
            if (path.points.isNotEmpty()) {
                val analysis = analyzer.analyzeMovement(path.points)
                val row = sheet.createRow(rowNum++)

                row.createCell(0).apply { setCellValue((pathIndex + 1).toDouble()); cellStyle = dataStyle }
                row.createCell(1).apply { setCellValue(analysis.direction.toString()); cellStyle = dataStyle }
                row.createCell(2).apply { setCellValue(analysis.velocity.toDouble()); cellStyle = dataStyle }
                row.createCell(3).apply { setCellValue(analysis.acceleration.toDouble()); cellStyle = dataStyle }
                row.createCell(4).apply { setCellValue(analysis.totalDistance.toDouble()); cellStyle = dataStyle }
                row.createCell(5).apply { setCellValue(analysis.peakVelocity.toDouble()); cellStyle = dataStyle }
                row.createCell(6).apply { setCellValue(analysis.averageBarSpeed.toDouble()); cellStyle = dataStyle }
            }
        }

        // Manual column sizing
        headers.forEachIndexed { index, _ ->
            sheet.setColumnWidth(index, 3500)
        }
    }

    private fun createRepAnalysisSheet(workbook: XSSFWorkbook, session: WorkoutSession, analyzer: BarPathAnalyzer) {
        val sheet = workbook.createSheet("Rep Analysis")
        val headerStyle = createHeaderStyle(workbook)
        val dataStyle = createDataStyle(workbook)

        var rowNum = 0

        // Headers
        val headerRow = sheet.createRow(rowNum++)
        val headers = arrayOf("Rep", "Range_of_Motion", "Duration(s)", "Total_Distance", "Path_Deviation", "Rep_Quality_Score")
        headers.forEachIndexed { index, header ->
            val cell = headerRow.createCell(index)
            cell.setCellValue(header)
            cell.cellStyle = headerStyle
        }

        // Analyze each rep
        session.paths.forEachIndexed { pathIndex, path ->
            if (path.points.isNotEmpty()) {
                val row = sheet.createRow(rowNum++)

                val rom = path.getVerticalRange()
                val duration = path.getDuration() / 1000f
                val totalDistance = path.getTotalDistance()
                val pathDeviation = calculatePathDeviation(path.points)
                val qualityScore = calculateRepQualityScore(rom, duration, totalDistance, pathDeviation)

                row.createCell(0).apply { setCellValue((pathIndex + 1).toDouble()); cellStyle = dataStyle }
                row.createCell(1).apply { setCellValue(rom.toDouble()); cellStyle = dataStyle }
                row.createCell(2).apply { setCellValue(duration.toDouble()); cellStyle = dataStyle }
                row.createCell(3).apply { setCellValue(totalDistance.toDouble()); cellStyle = dataStyle }
                row.createCell(4).apply { setCellValue(pathDeviation.toDouble()); cellStyle = dataStyle }
                row.createCell(5).apply { setCellValue(qualityScore.toDouble()); cellStyle = dataStyle }
            }
        }

        // Manual column sizing
        headers.forEachIndexed { index, _ ->
            sheet.setColumnWidth(index, 4000)
        }
    }

    private fun createStatisticsSheet(workbook: XSSFWorkbook, session: WorkoutSession, analyzer: BarPathAnalyzer) {
        val sheet = workbook.createSheet("Statistics")
        val headerStyle = createHeaderStyle(workbook)
        val dataStyle = createDataStyle(workbook)

        var rowNum = 0

        // Calculate comprehensive statistics
        val allPoints = session.paths.flatMap { it.points }
        val metrics = analyzer.calculateLiftingMetrics(session.paths)

        val stats = mapOf(
            "Total Data Points" to allPoints.size.toString(),
            "Average Points per Rep" to if (session.paths.isNotEmpty()) "${allPoints.size / session.paths.size}" else "0",
            "Session Duration (min)" to String.format("%.1f", (session.endTime - session.startTime) / 60000f),
            "Average Rep Duration (s)" to String.format("%.2f", metrics.averageRepTime),
            "Fastest Rep (s)" to String.format("%.2f", calculateFastestRep(session.paths)),
            "Slowest Rep (s)" to String.format("%.2f", calculateSlowestRep(session.paths)),
            "Best Range of Motion" to String.format("%.3f", calculateBestROM(session.paths)),
            "Average Range of Motion" to String.format("%.3f", metrics.averageRangeOfMotion),
            "Path Consistency" to String.format("%.1f%%", metrics.consistencyScore * 100),
            "Overall Quality Score" to String.format("%.1f%%", calculateOverallQuality(metrics) * 100)
        )

        // Title
        val titleRow = sheet.createRow(rowNum++)
        val titleCell = titleRow.createCell(0)
        titleCell.setCellValue("SESSION STATISTICS")
        titleCell.cellStyle = createTitleStyle(workbook)
        sheet.addMergedRegion(org.apache.poi.ss.util.CellRangeAddress(rowNum-1, rowNum-1, 0, 1))

        rowNum++ // Empty row

        // Statistics data
        stats.forEach { (key, value) ->
            val row = sheet.createRow(rowNum++)
            row.createCell(0).apply { setCellValue(key); cellStyle = headerStyle }
            row.createCell(1).apply { setCellValue(value); cellStyle = dataStyle }
        }

        // Manual column sizing
        sheet.setColumnWidth(0, 6000)
        sheet.setColumnWidth(1, 4000)
    }

    private fun calculatePathDeviation(points: List<PathPoint>): Float {
        if (points.isEmpty()) return 0f
        val centerX = points.map { it.x }.average().toFloat()
        return points.map { kotlin.math.abs(it.x - centerX) }.average().toFloat()
    }

    private fun calculateRepQualityScore(rom: Float, duration: Float, distance: Float, deviation: Float): Float {
        // Normalize and combine metrics for quality score (0-1)
        val romScore = kotlin.math.min(rom * 10f, 1f) // Assumes good ROM is ~0.1
        val durationScore = if (duration > 0) kotlin.math.min(2f / duration, 1f) else 0f // Optimal around 2s
        val efficiencyScore = if (distance > 0) kotlin.math.min(rom / distance, 1f) else 0f
        val consistencyScore = kotlin.math.max(0f, 1f - deviation * 20f)

        return (romScore * 0.3f + durationScore * 0.2f + efficiencyScore * 0.3f + consistencyScore * 0.2f)
    }

    private fun calculateFastestRep(paths: List<BarPath>): Float {
        return paths.mapNotNull { path ->
            if (path.points.isNotEmpty()) path.getDuration() / 1000f else null
        }.minOrNull() ?: 0f
    }

    private fun calculateSlowestRep(paths: List<BarPath>): Float {
        return paths.mapNotNull { path ->
            if (path.points.isNotEmpty()) path.getDuration() / 1000f else null
        }.maxOrNull() ?: 0f
    }

    private fun calculateBestROM(paths: List<BarPath>): Float {
        return paths.maxOfOrNull { it.getVerticalRange() } ?: 0f
    }

    private fun calculateOverallQuality(metrics: LiftingMetrics): Float {
        return (metrics.consistencyScore * 0.4f +
                kotlin.math.min(metrics.averageRangeOfMotion * 10f, 1f) * 0.3f +
                kotlin.math.max(0f, 1f - metrics.barPathDeviation * 10f) * 0.3f)
    }

    private fun saveWorkbookToFile(workbook: XSSFWorkbook, baseFileName: String): File {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val fileName = "${baseFileName}_$timestamp.xlsx"
        val file = File(context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), fileName)

        FileOutputStream(file).use { outputStream ->
            workbook.write(outputStream)
        }

        return file
    }

    // ANDROID-COMPATIBLE: Use XSSFCellStyle and avoid AWT dependencies
    private fun createTitleStyle(workbook: XSSFWorkbook): XSSFCellStyle {
        val style = workbook.createCellStyle()
        val font = workbook.createFont()
        font.bold = true
        font.fontHeightInPoints = 16
        style.setFont(font)
        style.alignment = HorizontalAlignment.CENTER
        return style
    }

    private fun createHeaderStyle(workbook: XSSFWorkbook): XSSFCellStyle {
        val style = workbook.createCellStyle()
        val font = workbook.createFont()
        font.bold = true
        font.fontHeightInPoints = 12
        style.setFont(font)
        style.fillForegroundColor = IndexedColors.GREY_25_PERCENT.getIndex()
        style.fillPattern = FillPatternType.SOLID_FOREGROUND
        style.borderBottom = BorderStyle.THIN
        style.borderTop = BorderStyle.THIN
        style.borderRight = BorderStyle.THIN
        style.borderLeft = BorderStyle.THIN
        return style
    }

    private fun createDataStyle(workbook: XSSFWorkbook): XSSFCellStyle {
        val style = workbook.createCellStyle()
        style.borderBottom = BorderStyle.THIN
        style.borderTop = BorderStyle.THIN
        style.borderRight = BorderStyle.THIN
        style.borderLeft = BorderStyle.THIN
        return style
    }

    private fun formatDuration(durationMs: Long): String {
        val minutes = durationMs / 60000
        val seconds = (durationMs % 60000) / 1000
        return "${minutes}m ${seconds}s"
    }

    /**
     * Share the generated report file
     */
    fun shareReport(file: File) {
        try {
            val uri = FileProvider.getUriForFile(context, PROVIDER_AUTHORITY, file)
            val intent = Intent(Intent.ACTION_SEND).apply {
                type = if (file.extension == "xlsx") {
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                } else {
                    "text/csv"
                }
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "Bar Path Analysis Report")
                putExtra(Intent.EXTRA_TEXT, "Bar path analysis report generated by Bar Path Detector app.")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }

            val chooser = Intent.createChooser(intent, "Share Report")
            chooser.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            context.startActivity(chooser)
        } catch (e: Exception) {
            Log.e(TAG, "Error sharing report: ${e.message}", e)
        }
    }
}