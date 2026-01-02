package com.programminghut.object_detection

import android.content.Intent
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private val TAG = "MainActivity"

    private lateinit var btn: Button
    private lateinit var imageView: ImageView
    private lateinit var bitmap: Bitmap
    private lateinit var interpreter: Interpreter
    private lateinit var labels: List<String>

    private var modelInputWidth = 1
    private var modelInputHeight = 1
    private var modelInputChannels = 3
    private var modelInputDataType: DataType = DataType.UINT8

    private val PICK_IMAGE_REQUEST = 101

    private val paint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.FILL
        strokeWidth = 5f
        textSize = 50f
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btn = findViewById(R.id.btn)
        imageView = findViewById(R.id.imaegView)

        // Load labels
        labels = try {
            FileUtil.loadLabels(this, "labels.txt")
        } catch (e: Exception) {
            Log.w(TAG, "labels.txt not found in assets, using fallback labels", e)
            listOf("class0", "class1")
        }

        // Load model and inspect input tensor
        try {
            val modelBuffer = FileUtil.loadMappedFile(this, "ssd_mobilenet_v1_1_metadata_1.tflite")
            interpreter = Interpreter(modelBuffer)
            Log.d(TAG, "TFLite interpreter created.")

            // Inspect input tensor 0
            val inputTensor = interpreter.getInputTensor(0)
            val inputShape = inputTensor.shape() // e.g. [1, height, width, 3]
            if (inputShape.size >= 4) {
                modelInputHeight = inputShape[1]
                modelInputWidth = inputShape[2]
                modelInputChannels = inputShape[3]
            } else if (inputShape.size == 2) {
                // e.g. [1, N] classification
                modelInputHeight = 1
                modelInputWidth = inputShape[1]
                modelInputChannels = 1
            }
            modelInputDataType = inputTensor.dataType()
            Log.i(TAG, "Model input shape: ${inputShape.joinToString()} datatype: $modelInputDataType")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model or inspect tensors", e)
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_LONG).show()
            return
        }

        val pickIntent = Intent(Intent.ACTION_GET_CONTENT).apply {
            type = "image/*"
        }

        btn.setOnClickListener {
            startActivityForResult(pickIntent, PICK_IMAGE_REQUEST)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode != PICK_IMAGE_REQUEST) return
        if (resultCode != RESULT_OK) {
            Log.w(TAG, "Image pick canceled or failed: resultCode=$resultCode")
            return
        }

        val uri = data?.data
        if (uri == null) {
            Log.e(TAG, "No image URI returned")
            Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            // Open stream and read image bounds first
            val input: InputStream? = contentResolver.openInputStream(uri)
            if (input == null) {
                Log.e(TAG, "Unable to open input stream for uri: $uri")
                Toast.makeText(this, "Unable to open selected image", Toast.LENGTH_SHORT).show()
                return
            }

            val boundsOptions = android.graphics.BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            android.graphics.BitmapFactory.decodeStream(input, null, boundsOptions)
            input.close()

            // Choose sample size to avoid OOM for huge images
            val maxSide = 2000
            var inSampleSize = 1
            val (width, height) = boundsOptions.outWidth to boundsOptions.outHeight
            if (width > 0 && height > 0) {
                var halfW = width / 2
                var halfH = height / 2
                while ((halfW / inSampleSize) >= maxSide && (halfH / inSampleSize) >= maxSide) {
                    inSampleSize *= 2
                }
            }

            // Re-open and decode actual bitmap with sampling
            val input2: InputStream? = contentResolver.openInputStream(uri)
            val decodeOptions = android.graphics.BitmapFactory.Options().apply {
                inSampleSize = inSampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
            }
            val decoded = android.graphics.BitmapFactory.decodeStream(input2, null, decodeOptions)
            input2?.close()

            if (decoded == null) {
                Log.e(TAG, "Failed to decode bitmap from uri: $uri")
                Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
                return
            }

            // Keep an ARGB_8888 copy to draw on and pass to TF
            bitmap = decoded.copy(Bitmap.Config.ARGB_8888, true)
            getPredictions()

        } catch (se: SecurityException) {
            Log.e(TAG, "Security exception loading image", se)
            Toast.makeText(this, "Permission denied reading image", Toast.LENGTH_LONG).show()
        } catch (oom: OutOfMemoryError) {
            Log.e(TAG, "OOM while loading image", oom)
            Toast.makeText(this, "Image too large to process", Toast.LENGTH_LONG).show()
        } catch (ex: Exception) {
            Log.e(TAG, "Error loading image", ex)
            Toast.makeText(this, "Failed to load image", Toast.LENGTH_SHORT).show()
        }
    }

    private fun buildImageProcessor(): ImageProcessor {
        // Resize to model input and, for FLOAT32 models, apply normalization
        val builder = ImageProcessor.Builder()
            .add(ResizeOp(modelInputHeight, modelInputWidth, ResizeOp.ResizeMethod.BILINEAR))

        if (modelInputDataType == DataType.FLOAT32) {
            // scale [0..255] -> [0..1]; if your model expects [-1,1] use NormalizeOp(127.5f,127.5f)
            builder.add(NormalizeOp(0f, 255f))
        }
        return builder.build()
    }

    private fun getPredictions() {
        try {
            if (!::interpreter.isInitialized) {
                Log.e(TAG, "Interpreter not initialized")
                Toast.makeText(this, "Model not loaded", Toast.LENGTH_SHORT).show()
                return
            }
            if (!::bitmap.isInitialized) {
                Log.e(TAG, "No bitmap loaded yet")
                Toast.makeText(this, "No image selected", Toast.LENGTH_SHORT).show()
                return
            }

            // Prepare TensorImage and load the bitmap pixels
            val tensorImage = if (modelInputDataType == DataType.FLOAT32) {
                TensorImage(DataType.FLOAT32)
            } else {
                TensorImage(DataType.UINT8)
            }
            tensorImage.load(bitmap) // MUST call before processing / getting buffer

            // Process (resize + normalize if needed)
            val imageProcessor = buildImageProcessor()
            val processedImage = imageProcessor.process(tensorImage)

            // Get ByteBuffer for interpreter input
            val inputBuffer: ByteBuffer = processedImage.buffer.rewind() as ByteBuffer
            inputBuffer.order(ByteOrder.nativeOrder())

            // Debug: check expected vs actual input bytes
            val inpTensor = interpreter.getInputTensor(0)
            val expectedBytes = inpTensor.numBytes()
            val actualBytes = inputBuffer.remaining()
            Log.d(TAG, "Input tensor expects $expectedBytes bytes, input buffer has $actualBytes bytes, dtype=${inpTensor.dataType()}")
            if (actualBytes != expectedBytes) {
                Log.e(TAG, "Input byte size mismatch! expected=$expectedBytes actual=$actualBytes")
                Toast.makeText(this, "Model input size mismatch. Check model input shape/type.", Toast.LENGTH_LONG).show()
                return
            }

            // Build output container dynamically based on output tensor shape
            val outTensor = interpreter.getOutputTensor(0)
            val outShape = outTensor.shape()
            val numOutputs = if (outShape.size >= 2) outShape[1] else outShape[0]
            val outputArray = Array(1) { FloatArray(numOutputs) }

            try {
                interpreter.run(inputBuffer, outputArray)
            } catch (e: Exception) {
                Log.e(TAG, "Interpreter run failed", e)
                Toast.makeText(this, "Model inference failed", Toast.LENGTH_LONG).show()
                return
            }

            Log.d(TAG, "raw output: ${outputArray[0].contentToString()}")

            // Find the predicted class
            val scores = outputArray[0]
            val bestIdx = scores.indices.maxByOrNull { scores[it] } ?: 0
            val confidence = scores[bestIdx]
            val label = labels.getOrNull(bestIdx) ?: "class_$bestIdx"

            // Draw label & confidence on a copy of the original image
            val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            val canvas = Canvas(mutable)
            paint.textSize = (mutable.height / 18f)

            val bgPaint = Paint().apply {
                color = Color.argb(160, 0, 0, 0)
                style = Paint.Style.FILL
            }

            val text = "$label  ${(confidence * 100).toInt()}%"
            val textPadding = 20f
            val textWidth = paint.measureText(text)
            val bgRect = RectF(0f, 0f, textWidth + textPadding * 2, paint.textSize + textPadding * 2)

            canvas.drawRect(bgRect, bgPaint)
            canvas.drawText(text, textPadding, paint.textSize + textPadding, paint)

            imageView.setImageBitmap(mutable)

        } catch (oom: OutOfMemoryError) {
            Log.e(TAG, "OOM during prediction", oom)
            Toast.makeText(this, "Not enough memory for inference", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error during prediction", e)
            Toast.makeText(this, "Error during prediction", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::interpreter.isInitialized) {
            interpreter.close()
        }
    }
}
