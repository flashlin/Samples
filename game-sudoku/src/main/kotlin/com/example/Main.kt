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

class SudokuLogic {
    private val board: Array<Array<Int>> = Array(9) { Array(9) { 0 } }
    
    fun generateNewGame() {
        // 初始化棋盤
        clearBoard()
        
        // 生成完整的數獨解
        generateFullSudoku()
        
        // 隨機移除一些數字來創建遊戲
        for (i in 0..8) {
            for (j in 0..8) {
                if (Random.nextDouble() < 0.7) {
                    board[i][j] = 0
                }
            }
        }
    }
    
    fun clearBoard() {
        for (i in 0..8) {
            for (j in 0..8) {
                board[i][j] = 0
            }
        }
    }
    
    private fun generateFullSudoku(): Boolean {
        // 依序填充每個 3x3 區塊
        for (blockRow in 0..2) {
            for (blockCol in 0..2) {
                if (!fillBlock(blockRow, blockCol)) {
                    clearBoard()
                    return generateFullSudoku()
                }
            }
        }
        return true
    }
    
    private fun fillBlock(blockRow: Int, blockCol: Int): Boolean {
        // 收集區塊內所有空格子的位置
        val emptyPositions = mutableListOf<Pair<Int, Int>>()
        for (i in 0..2) {
            for (j in 0..2) {
                val row = blockRow * 3 + i
                val col = blockCol * 3 + j
                if (board[row][col] == 0) {
                    emptyPositions.add(row to col)
                }
            }
        }
        
        // 如果區塊已經填滿，返回 true
        if (emptyPositions.isEmpty()) return true
        
        // 打亂空格子的順序
        val shuffledPositions = emptyPositions.shuffled()
        
        // 對每個空位嘗試填入 1-9
        for ((row, col) in shuffledPositions) {
            // 打亂 1-9 的數字順序
            val numbers = (1..9).shuffled()
            
            // 嘗試每個數字
            for (num in numbers) {
                // 先放入數字
                board[row][col] = num
                
                // 檢查是否合法
                if (isValid(row, col, num)) {
                    // 如果當前數字合法，繼續填充下一個位置
                    if (emptyPositions.size == 1 || fillBlock(blockRow, blockCol)) {
                        return true
                    }
                }
                
                // 如果不合法或後續填充失敗，還原並嘗試下一個數字
                board[row][col] = 0
            }
        }
        
        // 如果所有數字都試過了還是不行，返回 false
        return false
    }
    
    fun isValid(row: Int, col: Int, num: Int): Boolean {
        // 如果是空格，則視為有效
        if (num == 0) return true
        
        // 檢查行規則：同一行不能有重複數字
        for (x in 0..8) {
            if (x != col && board[row][x] == num) return false
        }
        
        // 檢查列規則：同一列不能有重複數字
        for (x in 0..8) {
            if (x != row && board[x][col] == num) return false
        }
        
        // 檢查區域規則：3x3區域內不能有重複數字
        val startRow = row - row % 3
        val startCol = col - col % 3
        for (i in 0..2) {
            for (j in 0..2) {
                if (i + startRow != row || j + startCol != col) {
                    if (board[i + startRow][j + startCol] == num) return false
                }
            }
        }
        
        return true
    }
    
    fun getBoard(): Array<Array<Int>> {
        return Array(9) { row ->
            Array(9) { col ->
                board[row][col]
            }
        }
    }
}

class SudokuApp : App(SudokuView::class)

class SudokuView : View() {
    private val sudokuLogic = SudokuLogic()
    private val sudokuBoard: Array<Array<Int>> = Array(9) { Array(9) { 0 } }
    private val fixedNumbers: Array<Array<Boolean>> = Array(9) { Array(9) { false } }
    private val userInputs: Array<Array<Int>> = Array(9) { Array(9) { 0 } }
    private var selectedCell: Pair<Int, Int>? = null
    private var cellSize = 50.0
    private val canvas = Canvas()
    
    override val root: VBox = vbox {
        setPrefSize(500.0, 600.0)
        spacing = 10.0
        padding = insets(20)

        val messageLabel = label("") {
            id = "successMessage"
            isVisible = false
        }

        // 創建一個容器來包裝 canvas，使其能夠自適應大小
        stackpane {
            // 設置最小尺寸
            minWidth = 300.0
            minHeight = 300.0
            
            // 設置首選尺寸
            prefWidth = 450.0
            prefHeight = 450.0
            
            // 綁定 canvas 大小到容器大小
            add(canvas.apply {
                widthProperty().bind(this@stackpane.widthProperty())
                heightProperty().bind(this@stackpane.heightProperty())
                
                // 當 canvas 大小改變時重新繪製
                widthProperty().addListener { _, _, _ -> drawBoard() }
                heightProperty().addListener { _, _, _ -> drawBoard() }
            })
        }
        
        canvas.setOnMouseClicked { event ->
            // 根據當前 canvas 大小計算 cellSize
            cellSize = minOf(canvas.width, canvas.height) / 9
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

    fun generateNewGame() {
        // 使用 SudokuLogic 生成新遊戲
        sudokuLogic.generateNewGame()
        val generatedBoard = sudokuLogic.getBoard()
        
        // 更新視圖數據
        for (i in 0..8) {
            for (j in 0..8) {
                sudokuBoard[i][j] = generatedBoard[i][j]
                fixedNumbers[i][j] = generatedBoard[i][j] != 0
                userInputs[i][j] = 0
            }
        }
    }

    private fun drawBoard() {
        val gc = canvas.graphicsContext2D
        gc.clearRect(0.0, 0.0, canvas.width, canvas.height)
        
        // 計算新的 cellSize
        cellSize = minOf(canvas.width, canvas.height) / 9
        
        // 繪製背景格子
        for (i in 0..8) {
            for (j in 0..8) {
                gc.lineWidth = 1.0
                gc.stroke = Color.BLACK
                gc.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize)
            }
        }

        // 繪製3x3區域的粗線
        gc.lineWidth = 3.0
        
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
                    gc.font = Font.font("Arial", FontWeight.BOLD, cellSize * 0.6) // 根據 cellSize 調整字體大小
                    gc.textAlign = TextAlignment.CENTER
                    gc.textBaseline = VPos.CENTER
                    
                    if (fixedNumbers[i][j]) {
                        gc.fill = Color.BLACK
                    } else {
                        gc.fill = if (sudokuLogic.isValid(i, j, value)) Color.BLUE else Color.RED
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

    private fun checkWinCondition() {
        var isComplete = true
        for (i in 0..8) {
            for (j in 0..8) {
                val currentNum = if (fixedNumbers[i][j]) sudokuBoard[i][j] else userInputs[i][j]
                if (currentNum == 0 || !sudokuLogic.isValid(i, j, currentNum)) {
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

    // 提供一個方法來獲取數獨板的內容
    fun getSudokuBoard(): Array<Array<Int>> {
        return Array(9) { row ->
            Array(9) { col ->
                sudokuBoard[row][col]
            }
        }
    }
}

fun main() {
    launch<SudokuApp>()
} 