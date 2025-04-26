package com.example.sudoku

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import androidx.core.content.ContextCompat
import com.example.sudoku.R

class SudokuGridView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val paint = Paint().apply {
        color = ContextCompat.getColor(context, R.color.white)
        strokeWidth = 3f
        style = Paint.Style.STROKE
    }

    private val thickPaint = Paint().apply {
        color = ContextCompat.getColor(context, R.color.white)
        strokeWidth = 9f
        style = Paint.Style.STROKE
    }

    private val textPaint = Paint().apply {
        color = ContextCompat.getColor(context, R.color.white)
        textSize = 48f
        textAlign = Paint.Align.CENTER
        isFakeBoldText = true
    }

    private val cellRect = Rect()
    private var cellSize = 0f
    private var board: Array<IntArray> = Array(9) { IntArray(9) }
    private var originalNumbers: Array<BooleanArray> = Array(9) { BooleanArray(9) }
    private var onCellClickListener: ((Int, Int) -> Unit)? = null

    fun setOnCellClickListener(listener: (Int, Int) -> Unit) {
        onCellClickListener = listener
    }

    fun setBoard(newBoard: Array<IntArray>, newOriginalNumbers: Array<BooleanArray>) {
        board = newBoard
        originalNumbers = newOriginalNumbers
        invalidate()
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        cellSize = (w / 9f).coerceAtMost(h / 9f)
        textPaint.textSize = cellSize * 0.6f
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_DOWN) {
            val col = (event.x / cellSize).toInt()
            val row = (event.y / cellSize).toInt()
            if (row in 0..8 && col in 0..8) {
                onCellClickListener?.invoke(row, col)
            }
            return true
        }
        return super.onTouchEvent(event)
    }

    private fun isValid(row: Int, col: Int, num: Int): Boolean {
        // 檢查行
        for (x in 0..8) {
            if (x != col && board[row][x] == num) {
                return false
            }
        }

        // 檢查列
        for (x in 0..8) {
            if (x != row && board[x][col] == num) {
                return false
            }
        }

        // 檢查 3x3 方塊
        val startRow = row - row % 3
        val startCol = col - col % 3
        for (i in 0..2) {
            for (j in 0..2) {
                if (i + startRow != row || j + startCol != col) {
                    if (board[i + startRow][j + startCol] == num) {
                        return false
                    }
                }
            }
        }

        return true
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // 繪製黑色背景
        canvas.drawColor(ContextCompat.getColor(context, R.color.black))

        // 繪製原始數字區域的暗灰色背景
        for (row in 0..8) {
            for (col in 0..8) {
                if (originalNumbers[row][col]) {
                    cellRect.set(
                        (col * cellSize).toInt(),
                        (row * cellSize).toInt(),
                        ((col + 1) * cellSize).toInt(),
                        ((row + 1) * cellSize).toInt()
                    )
                    canvas.drawRect(cellRect, Paint().apply {
                        color = ContextCompat.getColor(context, R.color.dark_gray)
                        style = Paint.Style.FILL
                    })
                }
            }
        }

        // 繪製所有單元格的邊框
        for (row in 0..8) {
            for (col in 0..8) {
                cellRect.set(
                    (col * cellSize).toInt(),
                    (row * cellSize).toInt(),
                    ((col + 1) * cellSize).toInt(),
                    ((row + 1) * cellSize).toInt()
                )
                canvas.drawRect(cellRect, paint)
            }
        }

        // 繪製 3x3 區塊的粗邊框
        for (blockRow in 0..2) {
            for (blockCol in 0..2) {
                val left = (blockCol * 3 * cellSize).toInt()
                val top = (blockRow * 3 * cellSize).toInt()
                val right = ((blockCol + 1) * 3 * cellSize).toInt()
                val bottom = ((blockRow + 1) * 3 * cellSize).toInt()
                canvas.drawRect(left.toFloat(), top.toFloat(), right.toFloat(), bottom.toFloat(), thickPaint)
            }
        }

        // 繪製數字
        for (row in 0..8) {
            for (col in 0..8) {
                val number = board[row][col]
                if (number != 0) {
                    val x = col * cellSize + cellSize / 2
                    val y = row * cellSize + cellSize / 2 + textPaint.textSize / 3
                    textPaint.color = if (originalNumbers[row][col]) {
                        ContextCompat.getColor(context, R.color.white)
                    } else {
                        if (isValid(row, col, number)) {
                            ContextCompat.getColor(context, R.color.white)
                        } else {
                            ContextCompat.getColor(context, R.color.red)
                        }
                    }
                    canvas.drawText(number.toString(), x, y, textPaint)
                }
            }
        }
    }
} 