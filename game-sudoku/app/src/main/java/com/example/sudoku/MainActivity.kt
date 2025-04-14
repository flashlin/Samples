package com.example.sudoku

import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.sudoku.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var game: SudokuGame
    private val cells = Array(9) { Array(9) { android.widget.TextView(this) } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        game = SudokuGame()
        setupGame()
        setupListeners()
    }

    private fun setupGame() {
        game.generateNewGame()
        updateBoard()
    }

    private fun updateBoard() {
        val board = game.getBoard()
        for (i in 0..8) {
            for (j in 0..8) {
                val cell = cells[i][j]
                cell.text = if (board[i][j] != 0) board[i][j].toString() else ""
                cell.textSize = 20f
                cell.gravity = android.view.Gravity.CENTER
                cell.setBackgroundResource(android.R.drawable.edit_text)
                cell.setOnClickListener {
                    showNumberPicker(i, j)
                }
                binding.sudokuGrid.addView(cell)
            }
        }
    }

    private fun showNumberPicker(row: Int, col: Int) {
        val numbers = arrayOf("1", "2", "3", "4", "5", "6", "7", "8", "9", "清除")
        android.app.AlertDialog.Builder(this)
            .setTitle("選擇數字")
            .setItems(numbers) { _, which ->
                if (which == 9) {
                    game.setNumber(row, col, 0)
                } else {
                    game.setNumber(row, col, which + 1)
                }
                updateBoard()
            }
            .show()
    }

    private fun setupListeners() {
        binding.newGameButton.setOnClickListener {
            setupGame()
        }

        binding.checkButton.setOnClickListener {
            if (game.checkSolution()) {
                Toast.makeText(this, "恭喜！答案正確！", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "答案不正確，請繼續努力！", Toast.LENGTH_SHORT).show()
            }
        }
    }
} 