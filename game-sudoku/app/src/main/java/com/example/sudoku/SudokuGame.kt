package com.example.sudoku

enum class Difficulty {
    EASY,
    MEDIUM,
    HARD
}

class SudokuGame {
    private var board = Array(9) { IntArray(9) }
    private var originalBoard = Array(9) { IntArray(9) }
    private var currentDifficulty = Difficulty.MEDIUM

    fun setDifficulty(difficulty: Difficulty) {
        currentDifficulty = difficulty
    }

    fun getDifficulty(): Difficulty {
        return currentDifficulty
    }

    fun generateNewGame() {
        // 清空棋盤
        for (i in 0..8) {
            for (j in 0..8) {
                board[i][j] = 0
                originalBoard[i][j] = 0
            }
        }

        // 生成一個有效的數獨謎題
        fillDiagonal()
        fillRemaining(0, 3)
        removeNumbers()
        
        // 保存原始數字
        for (i in 0..8) {
            for (j in 0..8) {
                originalBoard[i][j] = board[i][j]
            }
        }
    }

    private fun fillDiagonal() {
        for (i in 0..8 step 3) {
            fillBox(i, i)
        }
    }

    private fun fillBox(row: Int, col: Int) {
        var num: Int
        for (i in 0..2) {
            for (j in 0..2) {
                do {
                    num = (1..9).random()
                } while (!isValidInBox(row, col, num))
                board[row + i][col + j] = num
            }
        }
    }

    private fun isValidInBox(row: Int, col: Int, num: Int): Boolean {
        for (i in 0..2) {
            for (j in 0..2) {
                if (board[row + i][col + j] == num) {
                    return false
                }
            }
        }
        return true
    }

    private fun fillRemaining(i: Int, j: Int): Boolean {
        if (j >= 9) {
            return fillRemaining(i + 1, 0)
        }
        if (i >= 9) {
            return true
        }
        if (i < 3) {
            if (j < 3) {
                return fillRemaining(i, 3)
            }
        } else if (i < 6) {
            if (j == (i / 3) * 3) {
                return fillRemaining(i, j + 3)
            }
        } else {
            if (j == 6) {
                return fillRemaining(i + 1, 0)
            }
        }

        for (num in 1..9) {
            if (isValid(i, j, num)) {
                board[i][j] = num
                if (fillRemaining(i, j + 1)) {
                    return true
                }
                board[i][j] = 0
            }
        }
        return false
    }

    fun isValid(row: Int, col: Int, num: Int): Boolean {
        // 檢查行
        for (x in 0..8) {
            if (board[row][x] == num) {
                return false
            }
        }

        // 檢查列
        for (x in 0..8) {
            if (board[x][col] == num) {
                return false
            }
        }

        // 檢查 3x3 方塊
        val startRow = row - row % 3
        val startCol = col - col % 3
        for (i in 0..2) {
            for (j in 0..2) {
                if (board[i + startRow][j + startCol] == num) {
                    return false
                }
            }
        }

        return true
    }

    private fun removeNumbers() {
        // 根據難度等級決定要移除的數字數量
        val cellsToRemove = when (currentDifficulty) {
            Difficulty.EASY -> 35
            Difficulty.MEDIUM -> 45
            Difficulty.HARD -> 55
        }
        var count = cellsToRemove
        while (count != 0) {
            val cellId = (0..80).random()
            val i = cellId / 9
            val j = cellId % 9
            if (board[i][j] != 0) {
                board[i][j] = 0
                count--
            }
        }
    }

    fun getBoard(): Array<IntArray> {
        return board
    }

    fun setNumber(row: Int, col: Int, number: Int) {
        board[row][col] = number
    }

    fun checkSolution(): Boolean {
        // 檢查行
        for (i in 0..8) {
            val rowSet = mutableSetOf<Int>()
            for (j in 0..8) {
                if (board[i][j] == 0 || !rowSet.add(board[i][j])) {
                    return false
                }
            }
        }

        // 檢查列
        for (j in 0..8) {
            val colSet = mutableSetOf<Int>()
            for (i in 0..8) {
                if (board[i][j] == 0 || !colSet.add(board[i][j])) {
                    return false
                }
            }
        }

        // 檢查 3x3 方塊
        for (block in 0..8) {
            val blockSet = mutableSetOf<Int>()
            val startRow = (block / 3) * 3
            val startCol = (block % 3) * 3
            for (i in 0..2) {
                for (j in 0..2) {
                    if (board[startRow + i][startCol + j] == 0 || !blockSet.add(board[startRow + i][startCol + j])) {
                        return false
                    }
                }
            }
        }

        return true
    }

    fun isOriginalNumber(row: Int, col: Int): Boolean {
        return originalBoard[row][col] != 0
    }

    fun getOriginalBoard(): Array<IntArray> {
        return Array(9) { row ->
            IntArray(9) { col ->
                originalBoard[row][col]
            }
        }
    }

    fun restoreState(newBoard: Array<IntArray>, newOriginalBoard: Array<IntArray>) {
        for (i in 0..8) {
            for (j in 0..8) {
                board[i][j] = newBoard[i][j]
                originalBoard[i][j] = newOriginalBoard[i][j]
            }
        }
    }
} 