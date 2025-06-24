package com.example.atry.detector

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

/** Helper to convert a CameraX ImageProxy into a correctly‐rotated Bitmap. */
object BitmapUtils {

    fun imageProxyToBitmap(im: ImageProxy): Bitmap {
        // UV planes in NV21 order
        val yBuffer = im.planes[0].buffer
        val uBuffer = im.planes[1].buffer
        val vBuffer = im.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, im.width, im.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, im.width, im.height), 100, out)
        val bytes = out.toByteArray()

        // decode + rotate
        val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        val matrix = Matrix().apply { postRotate(im.imageInfo.rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
    }
}