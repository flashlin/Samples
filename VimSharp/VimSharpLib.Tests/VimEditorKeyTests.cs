using Xunit;
using NSubstitute;
using System;
using System.Collections.Generic;

namespace VimSharpLib.Tests
{
    public class VimEditorKeyTests
    {
        private IConsoleDevice _mockConsole;
        private VimEditor _editor;

        public VimEditorKeyTests()
        {
            _mockConsole = Substitute.For<IConsoleDevice>();
            _mockConsole.WindowWidth.Returns(80);
            _mockConsole.WindowHeight.Returns(25);
            _editor = new VimEditor(_mockConsole);
            _editor.Context.IsLineNumberVisible = false;
            _editor.Context.IsStatusBarVisible = false;
            _editor.Context.SetViewPort(0, 0, 10, 5);
        }

        [Fact]
        public void TestRightArrowKey()
        {
            // Arrange
            // 設置 ViewPort = 0, 0, 10, 5
            _editor.Context.SetViewPort(0, 0, 10, 5);

            // 加載文本 "Hello"
            _editor.OpenText("Hello");

            // 確保編輯器處於正常模式
            _editor.Mode = new VimNormalMode(_editor);

            // Act
            // 按下向右按鍵 6 次
            for (int i = 0; i < 6; i++)
            {
                _editor.Mode.PressKey(ConsoleKeyPress.RightArrow);
            }

            // Assert
            // 驗證 CursorX 應該是 4
            Assert.Equal(4, _editor.Context.CursorX);
        }

        [Fact]
        public void TestRightArrowKeyWithMultilineText()
        {
            // Arrange
            // 設置 ViewPort = 0, 0, 10, 5
            _editor.Context.SetViewPort(0, 0, 10, 5);

            // 加載多行文本 "Hello\r\nFlash"
            _editor.OpenText("Hello\r\nFlash");

            // 確保編輯器處於正常模式
            _editor.Mode = new VimNormalMode(_editor);

            // Act
            // 按下向右按鍵 6 次
            for (int i = 0; i < 6; i++)
            {
                _editor.Mode.PressKey(ConsoleKeyPress.RightArrow);
            }

            // Assert
            // 驗證 CursorX 應該是 4
            Assert.Equal(4, _editor.Context.CursorX);
            // 驗證 CursorY 應該是 0（第一行）
            Assert.Equal(0, _editor.Context.CursorY);
        }

        [Fact]
        public void TestRightArrowAndDownArrowKeys()
        {
            // Arrange
            // 設置 ViewPort = 0, 0, 10, 5
            _editor.Context.SetViewPort(0, 0, 10, 5);

            // 加載多行文本 "Hello\r\nHi"
            _editor.OpenText("Hello\r\nHi");

            // 確保編輯器處於正常模式
            _editor.Mode = new VimNormalMode(_editor);

            // Act
            // 按下向右按鍵 6 次
            for (int i = 0; i < 6; i++)
            {
                _editor.Mode.PressKey(ConsoleKeyPress.RightArrow);
            }

            // 按下向下按鍵 1 次
            _editor.Mode.PressKey(ConsoleKeyPress.DownArrow);

            // Assert
            // 驗證 CursorX 應該是 1
            Assert.Equal(1, _editor.Context.CursorX);
            // 驗證 CursorY 應該是 1（第二行）
            Assert.Equal(1, _editor.Context.CursorY);
        }

        [Fact]
        public void TestInitWithLineNumbers()
        {
            // Arrange
            // 設置 ViewPort = 0, 0, 10, 5
            _editor.Context.SetViewPort(0, 0, 10, 5);

            // 啟用行號顯示
            _editor.Context.IsLineNumberVisible = true;

            // 加載多行文本 "Hello\r\nHi"
            _editor.OpenText("Hello\r\nHi");

            // Assert
            // 驗證初始游標位置
            // 由於有兩行文本，行號寬度為 1 位數字 + 1 位空格 = 2
            int expectedLineNumberWidth = 2;

            // 驗證 CursorX 應該是 ViewPort.X + 行號寬度
            Assert.Equal(_editor.Context.ViewPort.X + expectedLineNumberWidth, _editor.Context.CursorX);

            // 驗證 CursorY 應該是 ViewPort.Y
            Assert.Equal(_editor.Context.ViewPort.Y, _editor.Context.CursorY);

            // Act
            // 按下向右按鍵 6 次
            for (int i = 0; i < 6; i++)
            {
                _editor.Mode.PressKey(ConsoleKeyPress.RightArrow);
            }

            // 按下向下按鍵 1 次
            _editor.Mode.PressKey(ConsoleKeyPress.DownArrow);

            // Assert
            // 驗證 CursorX 應該是 3
            Assert.Equal(3, _editor.Context.CursorX);
            // 驗證 CursorY 應該是 1（第二行）
            Assert.Equal(1, _editor.Context.CursorY);
        }

        [Fact]
        public void TestRightArrowKeyWithChineseCharacter()
        {
            // Arrange
            // 設置 ViewPort = 0, 0, 10, 5
            _editor.Context.SetViewPort(0, 0, 10, 5);

            // 加載包含中文字符的文本 "閃1"
            _editor.OpenText("閃1");

            // 確保編輯器處於正常模式
            _editor.Mode = new VimNormalMode(_editor);

            // Act
            // 按下向右按鍵 1 次
            _editor.Mode.PressKey(ConsoleKeyPress.RightArrow);

            // Assert
            // 驗證 CursorX 應該是 2（因為中文字符"閃"佔用兩個字符寬度）
            Assert.Equal(2, _editor.Context.CursorX);

            // 按下向左按鍵 1 次
            _editor.Mode.PressKey(ConsoleKeyPress.LeftArrow);

            // Assert
            // 驗證 CursorX 應該是 0
            Assert.Equal(0, _editor.Context.CursorX);
        }

        [Fact]
        public void TestDollarKeyMovesToEndOfLine()
        {
            // Arrange
            // 設置 ViewPort = 0, 0, 10, 5
            _editor.Context.SetViewPort(0, 0, 10, 5);

            // 加載文本 "Hello"
            _editor.OpenText("Hello");

            // 確保編輯器處於正常模式
            _editor.Mode = new VimNormalMode(_editor);

            // Act
            // 按下 '$' 按鍵
            _editor.Mode.PressKey(ConsoleKeyPress.DollarSign); // '$' 對應 Shift+4

            // Assert
            // 驗證 CursorX 應該是 4（"Hello" 的長度，游標位於最後一個字符上面）
            Assert.Equal(4, _editor.Context.CursorX);
        }

        [Fact]
        public void TestDollarKeyAndWaitForInput()
        {
            // Arrange
            // 加載文本 "Hello"
            _editor.OpenText("Hello");

            // Act
            // 設置 ReadKey 返回 $ 按鍵 (Shift+4)
            SetReadKey(ConsoleKeyPress.DollarSign);

            // Assert
            // 驗證 CursorX 應該是 4（"Hello" 的最後一個字符位置）
            Assert.Equal(4, _editor.Context.CursorX);

            // 確認模擬的 ReadKey 方法被調用了一次
            _mockConsole.Received(1).ReadKey(Arg.Any<bool>());
        }

        [Fact]
        public void TestDollarKeyWithChineseCharacters()
        {
            // Arrange
            // 加載包含中文字符的文本 "Hi 閃電"
            _editor.OpenText("Hi 閃電");

            // Act
            // 設置 ReadKey 返回 $ 按鍵 (Shift+4)
            SetReadKey(ConsoleKeyPress.DollarSign);

            // Assert
            // 驗證 CursorX 應該是 8
            // "Hi " 佔 3 個字符寬度，"閃" 佔 2 個字符寬度，"電" 佔 2 個字符寬度
            // 最後一個字, 原本是6, 但因為是中文字, 所以游標位置是 5
            Assert.Equal(5, _editor.Context.CursorX);

            // 確認模擬的 ReadKey 方法被調用了一次
            _mockConsole.Received(1).ReadKey(Arg.Any<bool>());
        }

        [Fact]
        public void TestDollarKeyWithLineNumbersVisible()
        {
            // Arrange
            // 設置行號可見
            _editor.Context.IsLineNumberVisible = true;
            _editor.Context.SetViewPort(1, 1, 10, 5);
            // 加載文本 "Hello"
            _editor.OpenText("Hello, World!");

            // 確保行號寬度被正確計算為 2
            Assert.Equal(2, _editor.Context.GetLineNumberWidth());

            // Act
            // 設置 ReadKey 返回 $ 按鍵 (Shift+4)
            SetReadKey(ConsoleKeyPress.DollarSign);

            // Assert
            // 驗證 CursorX 應該是 6
            // 因為行號寬度為 2，加上 "Hello" 的最後一個字符位置 12，加上ViewPort.X 1所以總共是 15
            Assert.Equal(15, _editor.Context.CursorX);
            Assert.Equal(13, _editor.GetActualTextX() + 1);

            // 確認模擬的 ReadKey 方法被調用了一次
            _mockConsole.Received(1).ReadKey(Arg.Any<bool>());
        }

        [Fact]
        public void TestDollarKeyWithLineNumbersAndStatusBarVisible()
        {
            // Arrange
            // 設置行號和狀態欄可見
            _editor.Context.IsLineNumberVisible = true;
            _editor.Context.IsStatusBarVisible = true;

            // 加載文本 "Hello"
            _editor.OpenText("Hello");

            // 確保行號寬度被正確計算為 2
            Assert.Equal(2, _editor.Context.GetLineNumberWidth());

            _editor.Render();
            // 驗證狀態欄顯示內容
            Assert.Equal(" Normal | Line: 1 | Col: 1 ", _editor.Context.StatusBar.ToString());

            // Act
            // 設置 ReadKey 返回 $ 按鍵 (Shift+4)
            SetReadKey(ConsoleKeyPress.DollarSign);

            // Assert
            // 驗證 CursorX 應該是 6
            // 因為行號寬度為 2，加上 "Hello" 的最後一個字符位置 4，所以總共是 6
            Assert.Equal(6, _editor.Context.CursorX);

            // 確認模擬的 ReadKey 方法被調用了一次
            _mockConsole.Received(1).ReadKey(Arg.Any<bool>());
        }

        [Fact]
        public void TestCustomViewPortInitialCursorPosition()
        {
            // Arrange
            // 設置自定義 ViewPort
            _editor.Context.SetViewPort(1, 1, 40, 5);

            // 加載文本 "Hello"
            _editor.OpenText("Hello");

            // Assert
            Assert.Equal(1, _editor.Context.CursorX);
            Assert.Equal(1, _editor.Context.CursorY);
        }

        [Fact]
        public void TestCustomViewPortDollarKey()
        {
            // Arrange
            _editor.Context.SetViewPort(1, 1, 40, 5);
            _editor.OpenText("Hello");

            // Act
            // 設置並按下 $ 按鍵 (Shift+4)，將游標移動到行尾
            SetReadKey(ConsoleKeyPress.DollarSign);

            // Assert
            // 驗證 $ 按鍵後游標位置 (應該在最後一個字符上)
            Assert.Equal(5, _editor.Context.CursorX); // 4 + ViewPort.X(1)
            Assert.Equal(1, _editor.Context.CursorY); // Should still be on the first line
        }

        [Fact]
        public void TestCustomViewPortAKey()
        {
            // Arrange
            _editor.Context.SetViewPort(1, 1, 40, 5);
            _editor.OpenText("Hello");
            SetReadKey(ConsoleKeyPress.DollarSign);

            // Act
            // 設置並按下 a 按鍵，切換到插入模式並將游標移到最後一個字符之後
            SetReadKey(ConsoleKeyPress.a);

            // Assert
            // 驗證 a 按鍵後游標位置和模式
            Assert.Equal(6, _editor.Context.CursorX); // 5 + ViewPort.X(1)
            Assert.Equal(1, _editor.Context.CursorY); // Should still be on the first line
            Assert.IsType<VimInsertMode>(_editor.Mode);
        }

        [Fact]
        public void TestCustomViewPortLeftArrow()
        {
            // Arrange
            _editor.Context.SetViewPort(1, 1, 40, 5);
            _editor.OpenText("Hello");
            SetReadKey(ConsoleKeyPress.DollarSign);
            SetReadKey(ConsoleKeyPress.a);

            // Act
            // 設置並按下左箭頭按鍵，將游標向左移動一位
            SetReadKey(ConsoleKeyPress.LeftArrow);

            // Assert
            // 驗證左箭頭按鍵後游標位置
            Assert.Equal(5, _editor.Context.CursorX);
            Assert.Equal(1, _editor.Context.CursorY); // Should still be on the first line
        }

        [Fact]
        public void TestCustomViewPortRightArrow()
        {
            // Arrange
            _editor.Context.SetViewPort(1, 1, 40, 5);
            _editor.OpenText("Hello");
            SetReadKey(ConsoleKeyPress.DollarSign);
            SetReadKey(ConsoleKeyPress.a);
            SetReadKey(ConsoleKeyPress.LeftArrow);

            // Act
            // 按下右鍵頭按鈕，將游標向右移動一位
            SetReadKey(ConsoleKeyPress.RightArrow);

            // Assert
            // 驗證右箭頭按鍵後游標位置
            Assert.Equal(6, _editor.Context.CursorX);
            Assert.Equal(1, _editor.Context.CursorY); // Should still be on the first line
        }

        [Fact]
        public void TestCustomViewPortInsertNumber()
        {
            // Arrange
            _editor.Context.SetViewPort(1, 1, 40, 5);
            _editor.OpenText("Hello");
            SetReadKey(ConsoleKeyPress.DollarSign);
            SetReadKey(ConsoleKeyPress.a);

            // Act
            // 設置並按下 1 按鍵，在文本末尾插入 1
            SetReadKey(ConsoleKeyPress.One);

            // Assert
            // 驗證插入 1 後的文本和游標位置
            Assert.Equal("Hello1", _editor.GetCurrentLine().ToString());
            Assert.Equal(7, _editor.Context.CursorX); // 6 + ViewPort.X(1)
            Assert.Equal(1, _editor.Context.CursorY); // Should still be on the first line
        }

        [Fact]
        public void TestCustomViewPortInsertSecondNumber()
        {
            // Arrange
            _editor.Context.SetViewPort(1, 1, 40, 5);
            _editor.OpenText("Hello");
            SetReadKey(ConsoleKeyPress.DollarSign);
            SetReadKey(ConsoleKeyPress.a);
            SetReadKey(ConsoleKeyPress.One);

            // Act
            // 設置並按下 2 按鍵，在文本末尾插入 2
            SetReadKey(ConsoleKeyPress.Two);

            // Assert
            // 驗證插入 2 後的文本和游標位置
            Assert.Equal("Hello12", _editor.GetCurrentLine().ToString());
            Assert.Equal(8, _editor.Context.CursorX); // 7 + ViewPort.X(1)
            Assert.Equal(1, _editor.Context.CursorY); // Should still be on the first line
        }

        [Fact]
        public void TestCustomViewPortBackspace()
        {
            // Arrange
            _editor.Context.SetViewPort(1, 1, 40, 5);
            _editor.OpenText("Hello");
            SetReadKey(ConsoleKeyPress.DollarSign);
            SetReadKey(ConsoleKeyPress.a);
            SetReadKey(ConsoleKeyPress.One);
            SetReadKey(ConsoleKeyPress.Two);

            // Act
            // 設置並按下 Backspace 按鍵，刪除最後一個字符
            SetReadKey(ConsoleKeyPress.Backspace);

            // Assert
            // 驗證刪除後的文本和游標位置
            Assert.Equal("Hello1", _editor.GetCurrentLine().ToString());
            Assert.Equal(7, _editor.Context.CursorX); // 6 + ViewPort.X(1)
            Assert.Equal(1, _editor.Context.CursorY); // Should still be on the first line
        }

        [Fact]
        public void TestCustomViewPortInsertLetter()
        {
            // Arrange
            _editor.Context.SetViewPort(1, 1, 40, 5);
            _editor.OpenText("Hello");
            SetReadKey(ConsoleKeyPress.DollarSign);
            SetReadKey(ConsoleKeyPress.a);
            SetReadKey(ConsoleKeyPress.One);
            SetReadKey(ConsoleKeyPress.Two);
            SetReadKey(ConsoleKeyPress.Backspace);

            // Act
            // 輸入 a 字母
            SetReadKey(ConsoleKeyPress.a);

            // Assert
            // 驗證插入 a 後的文本和游標位置
            Assert.Equal("Hello1a", _editor.GetCurrentLine().ToString());
            Assert.Equal(8, _editor.Context.CursorX); // 7 + ViewPort.X(1)
            Assert.Equal(1, _editor.Context.CursorY); // Should still be on the first line
        }

        [Fact]
        public void TestCustomViewPortLeftArrowAndDelete()
        {
            // Arrange
            _editor.Context.SetViewPort(1, 1, 40, 5);
            _editor.OpenText("Hello");
            SetReadKey(ConsoleKeyPress.DollarSign);
            SetReadKey(ConsoleKeyPress.a);
            SetReadKey(ConsoleKeyPress.One);
            SetReadKey(ConsoleKeyPress.Two);
            SetReadKey(ConsoleKeyPress.Backspace);
            SetReadKey(ConsoleKeyPress.a);

            // Act
            // 向左移動並刪除字符
            SetReadKey(ConsoleKeyPress.LeftArrow);
            SetReadKey(ConsoleKeyPress.Delete);

            // Assert
            // 驗證刪除後的文本和游標位置
            Assert.Equal("Hello1", _editor.GetCurrentLine().ToString());
            Assert.Equal(7, _editor.Context.CursorX); // 6 + ViewPort.X(1)
            Assert.Equal(1, _editor.Context.CursorY); // Should still be on the first line
        }

        [Fact]
        public void TestCustomViewPortEscapeToNormalMode()
        {
            // Arrange
            _editor.Context.SetViewPort(1, 1, 40, 5);
            _editor.OpenText("Hello");
            SetReadKey(ConsoleKeyPress.DollarSign);
            SetReadKey(ConsoleKeyPress.a);
            SetReadKey(ConsoleKeyPress.One);

            // Act
            // 設置並按下 Esc 按鍵，切換回普通模式
            SetReadKey(ConsoleKeyPress.Escape);

            // Assert
            // 驗證 Esc 按鍵後游標位置和模式
            Assert.Equal(6, _editor.Context.CursorX); // 5 + ViewPort.X(1)
            Assert.Equal(1, _editor.Context.CursorY); // Should still be on the first line
            Assert.IsType<VimNormalMode>(_editor.Mode);
        }

        [Fact]
        public void TestDeleteWordAndInsertText()
        {
            // Arrange
            // 設置自定義 ViewPort
            _editor.Context.SetViewPort(1, 1, 40, 5);

            // 加載文本 "Hello, World"
            _editor.OpenText("Hello, World");

            // 將游標移到 W 上
            // "Hello, " 長度為 7，加上 ViewPort.X(1)，游標位置應為 8
            _editor.Context.CursorX = 8;

            // 確保編輯器處於正常模式
            _editor.Mode = new VimNormalMode(_editor);

            // Act
            // 按下大寫 D 按鈕，刪除從當前位置到行尾
            SetReadKey(ConsoleKeyPress.D);
            Assert.Equal("Hello, ", _editor.GetCurrentLine().ToString());
            

            // 切換到插入模式
            SetReadKey(ConsoleKeyPress.i);

            // 輸入 123
            SetReadKey(ConsoleKeyPress.One);
            SetReadKey(ConsoleKeyPress.Two);
            SetReadKey(ConsoleKeyPress.Three);

            // Assert
            // 驗證當前行文本應該是 "Hello, 123"
            Assert.Equal("Hello, 123", _editor.GetCurrentLine().ToString());

            // 驗證游標位置
            Assert.Equal(11, _editor.Context.CursorX); // 8 + "123" 長度 = 11
            Assert.Equal(1, _editor.Context.CursorY); // 應該仍在第一行

            // 驗證當前模式是插入模式
            Assert.IsType<VimInsertMode>(_editor.Mode);
        }

        [Fact]
        public void TestInsertingNumericCharactersBeforeChineseCharacters()
        {
            // Arrange
            // 設置自定義 ViewPort
            _editor.Context.SetViewPort(1, 1, 40, 5);

            // 加載文本 "Hello, World"
            _editor.OpenText("Hello, 閃電!");

            // 將游標移到 ! 上
            _editor.Context.CursorX = 12;
            
            SetReadKey(ConsoleKeyPress.i);
            
            Assert.Equal(12, _editor.Context.CursorX);
            Assert.Equal(1, _editor.Context.CursorY);
            
            SetReadKey(ConsoleKeyPress.One);
            SetReadKey(ConsoleKeyPress.Two);
            SetReadKey(ConsoleKeyPress.Three);
            
            Assert.Equal("Hello, 閃電123!", _editor.GetCurrentLine().ToString());
        }

        [Fact]
        public void TestFindModeWithExclamationMark()
        {
            // Arrange
            _editor.Context.SetViewPort(1, 1, 40, 5);
            _editor.OpenText("Hello, World!\nExample");
            
            // 建立 screenBuffer
            var screenBuffer = new ColoredChar[25, 80];
            
            // 按下 'f' 進入尋找模式
            SetReadKey(ConsoleKeyPress.f);
            
            // 按下 '!' 尋找驚嘆號
            SetReadKey(ConsoleKeyPress.ExclamationMark);
            
            // Act
            // 呼叫 Render 方法
            _editor.Mode.Render(screenBuffer);
            
            // Assert
            // 驗證第一行的內容
            Assert.Equal('H', screenBuffer[1, 1].Char);
            Assert.Equal('e', screenBuffer[1, 2].Char);
            Assert.Equal('l', screenBuffer[1, 3].Char);
            Assert.Equal('l', screenBuffer[1, 4].Char);
            Assert.Equal('o', screenBuffer[1, 5].Char);
            Assert.Equal(',', screenBuffer[1, 6].Char);
            Assert.Equal(' ', screenBuffer[1, 7].Char);
            Assert.Equal('W', screenBuffer[1, 8].Char);
            Assert.Equal('o', screenBuffer[1, 9].Char);
            Assert.Equal('r', screenBuffer[1, 10].Char);
            Assert.Equal('l', screenBuffer[1, 11].Char);
            Assert.Equal('A', screenBuffer[1, 12].Char); // 標籤應該顯示在這裡
            Assert.Equal('!', screenBuffer[1, 13].Char);
        }

        private void SetReadKey(ConsoleKeyInfo keyInfo)
        {
            _mockConsole.ReadKey(Arg.Any<bool>()).Returns(keyInfo);
            // 調用 WaitForInput 方法，這將觸發模擬的 ReadKey 方法
            _editor.WaitForInput();
        }
    }
}