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
                // 設置基本格子線條寬度
                gc.lineWidth = 1.0
                gc.stroke = Color.BLACK
                gc.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize)
            }
        }

        // 繪製3x3區域的粗線
        gc.lineWidth = 5.0  // 將3x3區域的線條寬度設置為5.0
        
        // 繪製豎線
        for (i in 0..3) {
            gc.strokeLine(i * (cellSize * 3), 0.0, i * (cellSize * 3), canvas.height)
        }
        
        // 繪製橫線
        for (i in 0..3) {
            gc.strokeLine(0.0, i * (cellSize * 3), canvas.width, i * (cellSize * 3))
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
        // 初始化棋盤和標記
        for (i in 0..8) {
            for (j in 0..8) {
                sudokuBoard[i][j] = 0
                fixedNumbers[i][j] = false
                userInputs[i][j] = 0
            }
        }
        
        // 生成完整的數獨解
        generateFullSudoku()
        
        // 隨機移除一些數字來創建遊戲
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

    private fun generateFullSudoku(): Boolean {
        // 獲取未填充的 3x3 區塊
        val unfilledBlock = findUnfilledBlock()
        if (unfilledBlock == null) return true // 所有區塊都已填滿
        
        val (blockRow, blockCol) = unfilledBlock
        
        // 創建並打亂 1-9 的數字池
        val numbers = (1..9).toList().shuffled().toMutableList()
        
        // 嘗試填充當前區塊
        for (i in 0..2) {
            for (j in 0..2) {
                val row = blockRow * 3 + i
                val col = blockCol * 3 + j
                
                var numberPlaced = false
                for (num in numbers.toList()) {
                    if (isValid(row, col, num)) {
                        sudokuBoard[row][col] = num
                        numbers.remove(num)
                        numberPlaced = true
                        break
                    }
                }
                
                if (!numberPlaced) {
                    // 如果無法放置任何數字，回溯並清空當前區塊
                    clearBlock(blockRow, blockCol)
                    return false
                }
            }
        }
        
        // 遞迴填充下一個區塊
        return generateFullSudoku()
    }

    private fun findUnfilledBlock(): Pair<Int, Int>? {
        for (blockRow in 0..2) {
            for (blockCol in 0..2) {
                if (isBlockEmpty(blockRow, blockCol)) {
                    return Pair(blockRow, blockCol)
                }
            }
        }
        return null
    }

    private fun isBlockEmpty(blockRow: Int, blockCol: Int): Boolean {
        for (i in 0..2) {
            for (j in 0..2) {
                if (sudokuBoard[blockRow * 3 + i][blockCol * 3 + j] != 0) {
                    return false
                }
            }
        }
        return true
    }

    private fun clearBlock(blockRow: Int, blockCol: Int) {
        for (i in 0..2) {
            for (j in 0..2) {
                sudokuBoard[blockRow * 3 + i][blockCol * 3 + j] = 0
            }
        }
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