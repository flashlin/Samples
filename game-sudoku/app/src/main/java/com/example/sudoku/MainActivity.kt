package com.example.sudoku

import android.os.Bundle
import android.util.Log
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.sudoku.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var game: SudokuGame
    private var selectedNumber: Int? = null
    private var selectedCell: Pair<Int, Int>? = null

    companion object {
        private const val TAG = "SudokuGame"
        private const val KEY_BOARD = "board"
        private const val KEY_ORIGINAL_BOARD = "original_board"
        private const val KEY_SELECTED_NUMBER = "selected_number"
        private const val KEY_SELECTED_CELL_ROW = "selected_cell_row"
        private const val KEY_SELECTED_CELL_COL = "selected_cell_col"
        private const val KEY_DIFFICULTY = "difficulty"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        try {
            super.onCreate(savedInstanceState)
            Log.d(TAG, "onCreate started")
            
            binding = ActivityMainBinding.inflate(layoutInflater)
            setContentView(binding.root)
            Log.d(TAG, "View binding completed")
            
            game = SudokuGame()
            
            if (savedInstanceState != null) {
                // 恢復遊戲狀態
                val board = savedInstanceState.getSerializable(KEY_BOARD) as? Array<IntArray>
                val originalBoard = savedInstanceState.getSerializable(KEY_ORIGINAL_BOARD) as? Array<IntArray>
                selectedNumber = savedInstanceState.getInt(KEY_SELECTED_NUMBER, -1).takeIf { it != -1 }
                val row = savedInstanceState.getInt(KEY_SELECTED_CELL_ROW, -1)
                val col = savedInstanceState.getInt(KEY_SELECTED_CELL_COL, -1)
                if (row != -1 && col != -1) {
                    selectedCell = Pair(row, col)
                }
                
                if (board != null && originalBoard != null) {
                    game.restoreState(board, originalBoard)
                    updateBoard()
                } else {
                    setupGame()
                }
            } else {
                setupGame()
            }
            
            setupNumberButtons()
            setupListeners()
            setupDifficultySpinner()
            
            // 初始化按鈕顏色
            val numberButtons = listOf(
                binding.button1, binding.button2, binding.button3,
                binding.button4, binding.button5, binding.button6,
                binding.button7, binding.button8, binding.button9,
                binding.buttonClear
            )
            
            numberButtons.forEach { button ->
                button.setBackgroundColor(android.graphics.Color.WHITE)
                button.setTextColor(android.graphics.Color.BLACK)
            }
            
            Log.d(TAG, "Game setup completed")
        } catch (e: Exception) {
            Log.e(TAG, "Error in onCreate: ${e.message}", e)
            Toast.makeText(this, "發生錯誤: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        // 保存遊戲狀態
        outState.putSerializable(KEY_BOARD, game.getBoard())
        outState.putSerializable(KEY_ORIGINAL_BOARD, game.getOriginalBoard())
        outState.putInt(KEY_SELECTED_NUMBER, selectedNumber ?: -1)
        selectedCell?.let { (row, col) ->
            outState.putInt(KEY_SELECTED_CELL_ROW, row)
            outState.putInt(KEY_SELECTED_CELL_COL, col)
        }
        outState.putString(KEY_DIFFICULTY, game.getDifficulty().name)
    }

    private fun setupDifficultySpinner() {
        val difficulties = arrayOf("簡單", "中等", "高等")
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, difficulties)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        binding.difficultySpinner.adapter = adapter

        // 設置初始難度
        binding.difficultySpinner.setSelection(when (game.getDifficulty()) {
            Difficulty.EASY -> 0
            Difficulty.MEDIUM -> 1
            Difficulty.HARD -> 2
        })

        binding.difficultySpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: android.view.View?, position: Int, id: Long) {
                val newDifficulty = when (position) {
                    0 -> Difficulty.EASY
                    1 -> Difficulty.MEDIUM
                    2 -> Difficulty.HARD
                    else -> Difficulty.MEDIUM
                }
                game.setDifficulty(newDifficulty)
                setupGame()
            }

            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {
                // 不需要處理
            }
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
            val originalNumbers = Array(9) { row ->
                BooleanArray(9) { col ->
                    game.isOriginalNumber(row, col)
                }
            }
            binding.sudokuGrid.setBoard(board, originalNumbers)
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
                    // 清除當前選中的格子
                    selectedCell?.let { (row, col) ->
                        if (!game.isOriginalNumber(row, col)) {
                            game.setNumber(row, col, 0)
                            updateBoard()
                        }
                    }
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
            button.isEnabled = true
            if (index == 9) { // 清除按鈕
                if (selectedNumber == null) {
                    // 選中清除按鈕時變成藍色
                    button.setBackgroundColor(android.graphics.Color.parseColor("#2196F3"))
                    button.setTextColor(android.graphics.Color.WHITE)
                } else {
                    // 未選中時維持原本顏色
                    button.setBackgroundColor(android.graphics.Color.WHITE)
                    button.setTextColor(android.graphics.Color.BLACK)
                }
            } else {
                if (selectedNumber == index + 1) {
                    // 選中的數字按鈕變成藍色
                    button.setBackgroundColor(android.graphics.Color.parseColor("#2196F3"))
                    button.setTextColor(android.graphics.Color.WHITE)
                } else {
                    // 其他按鈕維持原本顏色
                    button.setBackgroundColor(android.graphics.Color.WHITE)
                    button.setTextColor(android.graphics.Color.BLACK)
                }
            }
        }
    }

    private fun setupListeners() {
        binding.sudokuGrid.setOnCellClickListener { row, col ->
            // 只有原始數字不能更改
            if (!game.isOriginalNumber(row, col)) {
                if (selectedNumber != null) {
                    game.setNumber(row, col, selectedNumber!!)
                    selectedNumber = null
                    updateBoard()
                    updateNumberButtonsState()
                } else {
                    // 如果沒有選中數字，則清除當前格子的數字
                    game.setNumber(row, col, 0)
                    updateBoard()
                }
            }
        }

        binding.newGameButton.setOnClickListener {
            setupGame()
            updateNumberButtonsState()
        }

        binding.checkButton.setOnClickListener {
            if (game.checkSolution()) {
                Toast.makeText(this, "恭喜！您已經完成數獨！", Toast.LENGTH_LONG).show()
            } else {
                Toast.makeText(this, "遊戲尚未完成，請繼續努力！", Toast.LENGTH_LONG).show()
            }
        }
    }
} 