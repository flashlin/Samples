package com.example.sudoku

import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.sudoku.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var game: SudokuGame
    private lateinit var cells: Array<Array<android.widget.TextView>>
    private var selectedNumber: Int? = null

    companion object {
        private const val TAG = "SudokuGame"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        try {
            super.onCreate(savedInstanceState)
            Log.d(TAG, "onCreate started")
            
            binding = ActivityMainBinding.inflate(layoutInflater)
            setContentView(binding.root)
            Log.d(TAG, "View binding completed")

            cells = Array(9) { Array(9) { android.widget.TextView(this) } }
            
            game = SudokuGame()
            setupGame()
            setupNumberButtons()
            setupListeners()
            Log.d(TAG, "Game setup completed")
        } catch (e: Exception) {
            Log.e(TAG, "Error in onCreate: ${e.message}", e)
            Toast.makeText(this, "發生錯誤: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun setupGame() {
        try {
            game.generateNewGame()
            updateBoard()
            Log.d(TAG, "Game board updated")
        } catch (e: Exception) {
            Log.e(TAG, "Error in setupGame: ${e.message}", e)
            Toast.makeText(this, "遊戲初始化失敗: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun updateBoard() {
        try {
            val board = game.getBoard()
            binding.sudokuGrid.removeAllViews()
            
            for (i in 0..8) {
                for (j in 0..8) {
                    val cell = cells[i][j]
                    cell.text = if (board[i][j] != 0) board[i][j].toString() else ""
                    cell.textSize = 24f
                    cell.gravity = android.view.Gravity.CENTER
                    
                    // 設置背景顏色
                    if (board[i][j] != 0 && game.isOriginalNumber(i, j)) {
                        // 原始數字使用暗灰色背景
                        cell.setBackgroundColor(android.graphics.Color.parseColor("#CCCCCC"))
                    } else {
                        // 可編輯的格子使用白色背景
                        cell.setBackgroundResource(android.R.drawable.edit_text)
                    }
                    
                    // 設置格子大小
                    val params = android.widget.GridLayout.LayoutParams().apply {
                        width = 0
                        height = 0
                        columnSpec = android.widget.GridLayout.spec(j, 1f)
                        rowSpec = android.widget.GridLayout.spec(i, 1f)
                        setMargins(2, 2, 2, 2)
                    }
                    cell.layoutParams = params
                    
                    cell.setOnClickListener {
                        // 只有原始數字不能更改
                        if (!game.isOriginalNumber(i, j)) {
                            if (selectedNumber != null) {
                                game.setNumber(i, j, selectedNumber!!)
                                updateBoard()
                                selectedNumber = null
                                updateNumberButtonsState()
                            }
                        }
                    }
                    binding.sudokuGrid.addView(cell)
                }
            }
            Log.d(TAG, "Board UI updated")
        } catch (e: Exception) {
            Log.e(TAG, "Error in updateBoard: ${e.message}", e)
            Toast.makeText(this, "更新遊戲板失敗: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun setupNumberButtons() {
        val numberButtons = listOf(
            binding.button1, binding.button2, binding.button3,
            binding.button4, binding.button5, binding.button6,
            binding.button7, binding.button8, binding.button9,
            binding.buttonClear
        )

        numberButtons.forEachIndexed { index, button ->
            button.setOnClickListener {
                if (index == 9) { // 清除按鈕
                    selectedNumber = null
                } else {
                    selectedNumber = index + 1
                }
                updateNumberButtonsState()
            }
        }
    }

    private fun updateNumberButtonsState() {
        val numberButtons = listOf(
            binding.button1, binding.button2, binding.button3,
            binding.button4, binding.button5, binding.button6,
            binding.button7, binding.button8, binding.button9,
            binding.buttonClear
        )

        numberButtons.forEachIndexed { index, button ->
            if (index == 9) { // 清除按鈕
                button.isEnabled = selectedNumber != null
            } else {
                button.isEnabled = selectedNumber != index + 1
            }
        }
    }

    private fun setupListeners() {
        try {
            binding.newGameButton.setOnClickListener {
                setupGame()
                selectedNumber = null
                updateNumberButtonsState()
            }

            binding.checkButton.setOnClickListener {
                if (game.checkSolution()) {
                    Toast.makeText(this, "恭喜！答案正確！", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "答案不正確，請繼續努力！", Toast.LENGTH_SHORT).show()
                }
            }
            Log.d(TAG, "Listeners setup completed")
        } catch (e: Exception) {
            Log.e(TAG, "Error in setupListeners: ${e.message}", e)
            Toast.makeText(this, "設定按鈕失敗: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }
} 