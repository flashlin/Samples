package com.example

import tornadofx.*
import javafx.scene.paint.Color
import javafx.scene.text.FontWeight
import javafx.geometry.Pos
import javafx.scene.control.Button
import javafx.scene.layout.GridPane
import javafx.scene.layout.VBox
import javafx.scene.layout.HBox
import javafx.scene.control.Label
import javafx.scene.canvas.Canvas
import javafx.scene.text.Font
import javafx.scene.text.TextAlignment
import javafx.geometry.VPos
import kotlin.random.Random

class SudokuApp : App(SudokuView::class)

class SudokuView : View() {
    private val sudokuBoard: Array<Array<Int>> = Array(9) { Array(9) { 0 } }
    private val fixedNumbers: Array<Array<Boolean>> = Array(9) { Array(9) { false } }
    private val userInputs: Array<Array<Int>> = Array(9) { Array(9) { 0 } }
    private var selectedCell: Pair<Int, Int>? = null
    private val cellSize = 50.0
    private val canvas = Canvas(450.0, 450.0)
    
    override val root: VBox = vbox {
        setPrefSize(500.0, 600.0)
        spacing = 10.0
        padding = insets(20)

        val messageLabel = label("") {
            id = "successMessage"
            isVisible = false
        }

        add(canvas)
        
        canvas.setOnMouseClicked { event ->
            val col = (event.x / cellSize).toInt()
            val row = (event.y / cellSize).toInt()
            if (row in 0..8 && col in 0..8) {
                selectedCell = row to col
                drawBoard()
            }
        }

        hbox {
            alignment = Pos.CENTER
            spacing = 10.0
            padding = insets(20)

            for (num in 1..9) {
                button(num.toString()) {
                    setPrefSize(40.0, 40.0)
                    
                    action {
                        selectedCell?.let { (row, col) ->
                            if (!fixedNumbers[row][col]) {
                                userInputs[row][col] = num
                                drawBoard()
                                checkWinCondition()
                            }
                        }
                    }
                }
            }
        }

        button("New Game") {
            setPrefSize(120.0, 40.0)
            
            action {
                generateNewGame()
                drawBoard()
                messageLabel.isVisible = false
            }
        }
    }

    init {
        generateNewGame()
        drawBoard()
    }

    private fun drawBoard() {
        val gc = canvas.graphicsContext2D
        gc.clearRect(0.0, 0.0, canvas.width, canvas.height)
        
        // 繪製背景格子
        for (i in 0..8) {
            for (j in 0..8) {
                gc.stroke = Color.BLACK
                gc.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize)
                
                // 繪製3x3區塊的粗線
                if (i % 3 == 0 && i != 0) {
                    gc.lineWidth = 2.0
                    gc.strokeLine(0.0, i * cellSize, canvas.width, i * cellSize)
                    gc.lineWidth = 1.0
                }
                if (j % 3 == 0 && j != 0) {
                    gc.lineWidth = 2.0
                    gc.strokeLine(j * cellSize, 0.0, j * cellSize, canvas.height)
                    gc.lineWidth = 1.0
                }
            }
        }

        // 繪製選中的格子
        selectedCell?.let { (row, col) ->
            gc.fill = Color.LIGHTBLUE.deriveColor(0.0, 1.0, 1.0, 0.3)
            gc.fillRect(col * cellSize, row * cellSize, cellSize, cellSize)
        }

        // 繪製數字
        for (i in 0..8) {
            for (j in 0..8) {
                val value = if (fixedNumbers[i][j]) sudokuBoard[i][j] else userInputs[i][j]
                if (value != 0) {
                    gc.font = Font.font("Arial", FontWeight.BOLD, 24.0)
                    gc.textAlign = TextAlignment.CENTER
                    gc.textBaseline = VPos.CENTER
                    
                    if (fixedNumbers[i][j]) {
                        gc.fill = Color.BLACK
                    } else {
                        gc.fill = if (isValid(i, j, value)) Color.BLUE else Color.RED
                    }
                    
                    gc.fillText(
                        value.toString(),
                        j * cellSize + cellSize / 2,
                        i * cellSize + cellSize / 2
                    )
                }
            }
        }
    }

    private fun generateNewGame() {
        for (i in 0..8) {
            for (j in 0..8) {
                sudokuBoard[i][j] = 0
                fixedNumbers[i][j] = false
                userInputs[i][j] = 0
            }
        }
        
        generateValidSudoku(0, 0)
        
        for (i in 0..8) {
            for (j in 0..8) {
                if (Random.nextDouble() < 0.7) {
                    sudokuBoard[i][j] = 0
                } else {
                    fixedNumbers[i][j] = true
                }
            }
        }
    }

    private fun generateValidSudoku(row: Int, col: Int): Boolean {
        if (col == 9) {
            return generateValidSudoku(row + 1, 0)
        }
        if (row == 9) {
            return true
        }
        
        val numbers = (1..9).shuffled()
        for (num in numbers) {
            if (isValid(row, col, num)) {
                sudokuBoard[row][col] = num
                if (generateValidSudoku(row, col + 1)) {
                    return true
                }
                sudokuBoard[row][col] = 0
            }
        }
        return false
    }

    private fun isValid(row: Int, col: Int, num: Int): Boolean {
        // 如果是空格，則視為有效
        if (num == 0) return true
        
        // 檢查行規則：同一行不能有重複數字
        for (x in 0..8) {
            if (x != col) {
                val valueToCheck = if (fixedNumbers[row][x]) sudokuBoard[row][x] else userInputs[row][x]
                if (valueToCheck == num) return false
            }
        }
        
        // 檢查列規則：同一列不能有重複數字
        for (x in 0..8) {
            if (x != row) {
                val valueToCheck = if (fixedNumbers[x][col]) sudokuBoard[x][col] else userInputs[x][col]
                if (valueToCheck == num) return false
            }
        }
        
        // 檢查區域規則：3x3區域內不能有重複數字
        val startRow = row - row % 3
        val startCol = col - col % 3
        for (i in 0..2) {
            for (j in 0..2) {
                if (i + startRow != row || j + startCol != col) {
                    val currentRow = i + startRow
                    val currentCol = j + startCol
                    val valueToCheck = if (fixedNumbers[currentRow][currentCol]) 
                        sudokuBoard[currentRow][currentCol] 
                    else 
                        userInputs[currentRow][currentCol]
                    if (valueToCheck == num) return false
                }
            }
        }
        
        return true
    }

    private fun checkWinCondition() {
        var isComplete = true
        for (i in 0..8) {
            for (j in 0..8) {
                val currentNum = if (fixedNumbers[i][j]) sudokuBoard[i][j] else userInputs[i][j]
                if (currentNum == 0 || !isValid(i, j, currentNum)) {
                    isComplete = false
                    break
                }
            }
        }
        
        if (isComplete) {
            root.lookup("#successMessage").apply {
                (this as Label).text = "成功！"
                style = "-fx-background-color: blue; -fx-text-fill: white; -fx-padding: 10;"
                isVisible = true
            }
        }
    }
}

fun main() {
    launch<SudokuApp>() 
} 