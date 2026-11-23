
package com.sani.potholedetector

import android.graphics.Bitmap
import android.opengl.EGL14
import android.opengl.EGLExt
import android.opengl.GLES20
import android.opengl.GLUtils
import android.view.Surface
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

class EglFrameRenderer(
    private val inputSurface: Surface,
    private val width: Int,
    private val height: Int
) {

    private var display = EGL14.EGL_NO_DISPLAY
    private var context = EGL14.EGL_NO_CONTEXT
    private var eglSurface = EGL14.EGL_NO_SURFACE

    private var textureId = 0
    private var program = 0
    private var textureInitialized = false

    private val vertexShaderCode = """
        attribute vec4 aPosition;
        attribute vec2 aTexCoord;
        varying vec2 vTexCoord;

        void main() {
            gl_Position = aPosition;
            vTexCoord = aTexCoord;
        }
    """.trimIndent()

    private val fragmentShaderCode = """
        precision mediump float;
        varying vec2 vTexCoord;
        uniform sampler2D uTexture;

        void main() {
            gl_FragColor = texture2D(uTexture, vTexCoord);
        }
    """.trimIndent()

    private val vertices = floatArrayOf(
        -1f, -1f,
        1f, -1f,
        -1f,  1f,
        1f,  1f
    )

    private val textureCoordinates = floatArrayOf(
        0f, 1f,
        1f, 1f,
        0f, 0f,
        1f, 0f
    )

    private fun floatBuffer(values: FloatArray): FloatBuffer {
        return ByteBuffer.allocateDirect(values.size * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
            .put(values)
            .apply { position(0) }
    }

    private val vertexBuffer = floatBuffer(vertices)
    private val textureBuffer = floatBuffer(textureCoordinates)

    fun initialize() {
        display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)

        check(display != EGL14.EGL_NO_DISPLAY) {
            "Unable to get EGL display"
        }

        val version = IntArray(2)

        check(
            EGL14.eglInitialize(
                display,
                version,
                0,
                version,
                1
            )
        ) {
            "Unable to initialize EGL"
        }

        val configAttributes = intArrayOf(
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_RENDERABLE_TYPE,
            EGL14.EGL_OPENGL_ES2_BIT,
            0x3142, 1,
            EGL14.EGL_NONE
        )

        val configs = arrayOfNulls<android.opengl.EGLConfig>(1)
        val numberOfConfigs = IntArray(1)

        check(
            EGL14.eglChooseConfig(
                display,
                configAttributes,
                0,
                configs,
                0,
                1,
                numberOfConfigs,
                0
            )
        ) {
            "Unable to choose EGL configuration"
        }

        val config = configs[0]
            ?: throw IllegalStateException("No EGL configuration found")

        val contextAttributes = intArrayOf(
            EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
            EGL14.EGL_NONE
        )

        context = EGL14.eglCreateContext(
            display,
            config,
            EGL14.EGL_NO_CONTEXT,
            contextAttributes,
            0
        )

        check(context != EGL14.EGL_NO_CONTEXT) {
            "Unable to create EGL context"
        }

        eglSurface = EGL14.eglCreateWindowSurface(
            display,
            config,
            inputSurface,
            intArrayOf(EGL14.EGL_NONE),
            0
        )

        check(eglSurface != EGL14.EGL_NO_SURFACE) {
            "Unable to create EGL window surface"
        }

        makeCurrent()

        program = createProgram(
            vertexShaderCode,
            fragmentShaderCode
        )

        val textures = IntArray(1)

        GLES20.glGenTextures(1, textures, 0)
        textureId = textures[0]

        GLES20.glBindTexture(
            GLES20.GL_TEXTURE_2D,
            textureId
        )

        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_MIN_FILTER,
            GLES20.GL_LINEAR
        )

        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_MAG_FILTER,
            GLES20.GL_LINEAR
        )

        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_WRAP_S,
            GLES20.GL_CLAMP_TO_EDGE
        )

        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_WRAP_T,
            GLES20.GL_CLAMP_TO_EDGE
        )
    }

    private fun makeCurrent() {
        check(
            EGL14.eglMakeCurrent(
                display,
                eglSurface,
                eglSurface,
                context
            )
        ) {
            "Unable to make EGL context current"
        }
    }

    fun drawFrame(bitmap: Bitmap, presentationTimeUs: Long) {
        check(!bitmap.isRecycled) {
            "Cannot render a recycled bitmap"
        }

        require(bitmap.width == width && bitmap.height == height) {
            "Bitmap dimensions must match encoder dimensions"
        }

        makeCurrent()

        GLES20.glViewport(0, 0, width, height)

        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        GLES20.glBindTexture(
            GLES20.GL_TEXTURE_2D,
            textureId
        )

        if (!textureInitialized) {
            GLUtils.texImage2D(
                GLES20.GL_TEXTURE_2D,
                0,
                bitmap,
                0
            )
            textureInitialized = true
        } else {
            GLUtils.texSubImage2D(
                GLES20.GL_TEXTURE_2D,
                0,
                0,
                0,
                bitmap
            )
        }

        GLES20.glUseProgram(program)

        val positionLocation = GLES20.glGetAttribLocation(
            program,
            "aPosition"
        )

        val textureLocation = GLES20.glGetAttribLocation(
            program,
            "aTexCoord"
        )

        val samplerLocation = GLES20.glGetUniformLocation(
            program,
            "uTexture"
        )

        GLES20.glEnableVertexAttribArray(positionLocation)

        GLES20.glVertexAttribPointer(
            positionLocation,
            2,
            GLES20.GL_FLOAT,
            false,
            0,
            vertexBuffer
        )

        GLES20.glEnableVertexAttribArray(textureLocation)

        GLES20.glVertexAttribPointer(
            textureLocation,
            2,
            GLES20.GL_FLOAT,
            false,
            0,
            textureBuffer
        )

        GLES20.glUniform1i(samplerLocation, 0)

        GLES20.glDrawArrays(
            GLES20.GL_TRIANGLE_STRIP,
            0,
            4
        )

        EGLExt.eglPresentationTimeANDROID(
            display,
            eglSurface,
            presentationTimeUs * 1000L
        )

        check(
            EGL14.eglSwapBuffers(display, eglSurface)
        ) {
            "Unable to submit frame to encoder"
        }

        GLES20.glDisableVertexAttribArray(positionLocation)
        GLES20.glDisableVertexAttribArray(textureLocation)
    }

    private fun createProgram(
        vertexCode: String,
        fragmentCode: String
    ): Int {

        val vertexShader = loadShader(
            GLES20.GL_VERTEX_SHADER,
            vertexCode
        )

        val fragmentShader = loadShader(
            GLES20.GL_FRAGMENT_SHADER,
            fragmentCode
        )

        val shaderProgram = GLES20.glCreateProgram()

        GLES20.glAttachShader(shaderProgram, vertexShader)
        GLES20.glAttachShader(shaderProgram, fragmentShader)

        GLES20.glLinkProgram(shaderProgram)

        val linkStatus = IntArray(1)

        GLES20.glGetProgramiv(
            shaderProgram,
            GLES20.GL_LINK_STATUS,
            linkStatus,
            0
        )

        check(linkStatus[0] == GLES20.GL_TRUE) {
            "Unable to link OpenGL program: " +
                    GLES20.glGetProgramInfoLog(shaderProgram)
        }

        GLES20.glDeleteShader(vertexShader)
        GLES20.glDeleteShader(fragmentShader)

        return shaderProgram
    }

    private fun loadShader(type: Int, code: String): Int {
        val shader = GLES20.glCreateShader(type)

        GLES20.glShaderSource(shader, code)
        GLES20.glCompileShader(shader)

        val compileStatus = IntArray(1)

        GLES20.glGetShaderiv(
            shader,
            GLES20.GL_COMPILE_STATUS,
            compileStatus,
            0
        )

        check(compileStatus[0] == GLES20.GL_TRUE) {
            "Shader compilation failed: " +
                    GLES20.glGetShaderInfoLog(shader)
        }

        return shader
    }

    fun release() {
        if (display != EGL14.EGL_NO_DISPLAY) {
            if (context != EGL14.EGL_NO_CONTEXT &&
                eglSurface != EGL14.EGL_NO_SURFACE
            ) {
                makeCurrent()

                if (textureId != 0) {
                    GLES20.glDeleteTextures(
                        1,
                        intArrayOf(textureId),
                        0
                    )
                }

                if (program != 0) {
                    GLES20.glDeleteProgram(program)
                }
            }

            EGL14.eglMakeCurrent(
                display,
                EGL14.EGL_NO_SURFACE,
                EGL14.EGL_NO_SURFACE,
                EGL14.EGL_NO_CONTEXT
            )

            if (eglSurface != EGL14.EGL_NO_SURFACE) {
                EGL14.eglDestroySurface(display, eglSurface)
            }

            if (context != EGL14.EGL_NO_CONTEXT) {
                EGL14.eglDestroyContext(display, context)
            }

            EGL14.eglTerminate(display)
        }

        display = EGL14.EGL_NO_DISPLAY
        context = EGL14.EGL_NO_CONTEXT
        eglSurface = EGL14.EGL_NO_SURFACE

        textureId = 0
        program = 0
        textureInitialized = false
    }
}
