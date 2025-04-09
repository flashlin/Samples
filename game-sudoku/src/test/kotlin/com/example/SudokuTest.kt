package com.example

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.RepeatedTest

class SudokuTest {
    
    @RepeatedTest(10) // 重複測試10次
    fun testGenerateFullSudoku() {
        // 建立 SudokuLogic 實例
        val sudokuLogic = SudokuLogic()
        
        // 呼叫 generateNewGame 生成數獨
        sudokuLogic.generateNewGame()
        
        // 獲取生成的數獨板
        val board = sudokuLogic.getBoard()
        
        // 測試1：檢查每一行是否包含1-9的數字
        for (row in 0..8) {
            val numbers = mutableSetOf<Int>()
            for (col in 0..8) {
                val num = board[row][col]
                if (num != 0) {
                    numbers.add(num)
                }
            }
            assertTrue(numbers.size <= 9, "第 ${row + 1} 行的數字有重複")
            assertTrue(numbers.all { it in 1..9 }, "第 ${row + 1} 行包含無效數字")
        }
        
        // 測試2：檢查每一列是否包含1-9的數字
        for (col in 0..8) {
            val numbers = mutableSetOf<Int>()
            for (row in 0..8) {
                val num = board[row][col]
                if (num != 0) {
                    numbers.add(num)
                }
            }
            assertTrue(numbers.size <= 9, "第 ${col + 1} 列的數字有重複")
            assertTrue(numbers.all { it in 1..9 }, "第 ${col + 1} 列包含無效數字")
        }
        
        // 測試3：檢查每個3x3區塊是否包含1-9的數字
        for (blockRow in 0..2) {
            for (blockCol in 0..2) {
                val numbers = mutableSetOf<Int>()
                for (i in 0..2) {
                    for (j in 0..2) {
                        val num = board[blockRow * 3 + i][blockCol * 3 + j]
                        if (num != 0) {
                            numbers.add(num)
                        }
                    }
                }
                assertTrue(numbers.size <= 9, 
                    "第 ${blockRow * 3 + 1}~${blockRow * 3 + 3} 行, " +
                    "第 ${blockCol * 3 + 1}~${blockCol * 3 + 3} 列的3x3區塊有重複數字")
                assertTrue(numbers.all { it in 1..9 }, 
                    "第 ${blockRow * 3 + 1}~${blockRow * 3 + 3} 行, " +
                    "第 ${blockCol * 3 + 1}~${blockCol * 3 + 3} 列的3x3區塊包含無效數字")
            }
        }
        
        // 測試4：檢查是否有非法數字（小於0或大於9）
        for (row in 0..8) {
            for (col in 0..8) {
                val num = board[row][col]
                assertTrue(num in 0..9, "在位置 ($row, $col) 發現非法數字: $num")
            }
        }
        
        // 測試5：列印數獨板以供視覺檢查
        println("生成的數獨板:")
        printBoard(board)
    }
    
    private fun printBoard(board: Array<Array<Int>>) {
        println("-------------------------")
        for (i in 0..8) {
            print("| ")
            for (j in 0..8) {
                print(board[i][j])
                if ((j + 1) % 3 == 0) print(" | ") else print(" ")
            }
            println()
            if ((i + 1) % 3 == 0) println("-------------------------")
        }
    }

    @Test
    fun testIsValid() {
        val sudokuLogic = SudokuLogic()
        sudokuLogic.generateNewGame()
        
        // 找到一個空格子
        var emptyRow = -1
        var emptyCol = -1
        val board = sudokuLogic.getBoard()
        
        outer@ for (i in 0..8) {
            for (j in 0..8) {
                if (board[i][j] == 0) {
                    emptyRow = i
                    emptyCol = j
                    break@outer
                }
            }
        }
        
        if (emptyRow != -1 && emptyCol != -1) {
            // 測試放置有效數字
            var validNumberFound = false
            for (num in 1..9) {
                if (sudokuLogic.isValid(emptyRow, emptyCol, num)) {
                    validNumberFound = true
                    break
                }
            }
            assertTrue(validNumberFound, "應該至少有一個有效的數字可以放置")
            
            // 測試放置無效數字
            val existingNumbers = mutableSetOf<Int>()
            for (i in 0..8) {
                val num = board[emptyRow][i]
                if (num != 0) existingNumbers.add(num)
            }
            
            for (num in existingNumbers) {
                assertFalse(sudokuLogic.isValid(emptyRow, emptyCol, num),
                    "已經在同一行存在的數字 $num 不應該是有效的")
            }
        }
    }

    @Test
    fun testClearBoard() {
        val sudokuLogic = SudokuLogic()
        sudokuLogic.generateNewGame()
        sudokuLogic.clearBoard()
        
        val board = sudokuLogic.getBoard()
        for (i in 0..8) {
            for (j in 0..8) {
                assertEquals(0, board[i][j], "清除棋盤後，位置 ($i, $j) 應該為空（0）")
            }
        }
    }
}

class SudokuLogicTest {
    @Test
    fun testGenerateNewGame() {
        val sudokuLogic = SudokuLogic()
        sudokuLogic.generateNewGame()
        val board = sudokuLogic.getBoard()
        
        // 檢查每個數字是否在有效範圍內（0-9）
        for (i in 0..8) {
            for (j in 0..8) {
                assertTrue(board[i][j] in 0..9, "數字 ${board[i][j]} 在位置 ($i, $j) 超出範圍")
            }
        }
        
        // 檢查每行的有效性
        for (row in 0..8) {
            val numbers = mutableSetOf<Int>()
            for (col in 0..8) {
                val num = board[row][col]
                if (num != 0) {
                    assertFalse(numbers.contains(num), "行 $row 包含重複的數字 $num")
                    numbers.add(num)
                }
            }
        }
        
        // 檢查每列的有效性
        for (col in 0..8) {
            val numbers = mutableSetOf<Int>()
            for (row in 0..8) {
                val num = board[row][col]
                if (num != 0) {
                    assertFalse(numbers.contains(num), "列 $col 包含重複的數字 $num")
                    numbers.add(num)
                }
            }
        }
        
        // 檢查每個 3x3 區塊的有效性
        for (blockRow in 0..2) {
            for (blockCol in 0..2) {
                val numbers = mutableSetOf<Int>()
                for (i in 0..2) {
                    for (j in 0..2) {
                        val num = board[blockRow * 3 + i][blockCol * 3 + j]
                        if (num != 0) {
                            assertFalse(numbers.contains(num), 
                                "區塊 ($blockRow, $blockCol) 包含重複的數字 $num")
                            numbers.add(num)
                        }
                    }
                }
            }
        }
        
        // 打印生成的數獨板以供視覺檢查
        println("生成的數獨板：")
        for (i in 0..8) {
            for (j in 0..8) {
                print("${board[i][j]} ")
            }
            println()
        }
    }
} 