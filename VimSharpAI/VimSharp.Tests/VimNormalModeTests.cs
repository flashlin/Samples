using System;
using Xunit;
using NSubstitute;
using VimSharpLib;
using System.Collections.Generic;

namespace VimSharp.Tests
{
    public class VimNormalModeTests
    {
        private IConsoleDevice _mockConsole;
        private VimEditor _editor;
        private VimNormalMode _normalMode;

        public VimNormalModeTests()
        {
            // 設置模擬的控制台裝置
            _mockConsole = Substitute.For<IConsoleDevice>();
            _mockConsole.WindowWidth.Returns(80);
            _mockConsole.WindowHeight.Returns(25);

            // 創建 VimEditor 實例
            _editor = new VimEditor(_mockConsole);

            // 創建並設置 VimNormalMode
            _normalMode = new VimNormalMode();
            _normalMode.Instance = _editor;
            _editor.Mode = _normalMode;

            // 設置一個簡單的測試文本
            _editor.Texts.Clear();
            _editor.Texts.Add(new ConsoleText("This is a test line."));
            _editor.Texts.Add(new ConsoleText("Another test line for vim commands."));
            _editor.Texts.Add(new ConsoleText("Third line."));
        }

        [Fact]
        public void TestMoveCursorToStartOfLine()
        {
            // 設置初始游標位置（非行首）
            _editor.CursorX = 5;
            _editor.CursorY = 1;

            // 模擬按下 '^' 鍵 (Shift+6, 對應 ConsoleKey.D6)
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('^', ConsoleKey.D6, true, false, false));

            // 執行按鍵處理
            _normalMode.WaitForInput();

            // 驗證游標是否移動到行首
            Assert.Equal(0, _editor.CursorX);
            Assert.Equal(1, _editor.CursorY); // Y 應該不變
        }

        [Fact]
        public void TestMoveCursorToEndOfLine()
        {
            // 設置初始游標位置（非行尾）
            _editor.CursorX = 5;
            _editor.CursorY = 1;

            // 獲取第二行的長度以進行驗證
            int expectedX = _editor.Texts[1].Width - 1;

            // 模擬按下 '$' 鍵 (Shift+4, 對應 ConsoleKey.D4)
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('$', ConsoleKey.D4, true, false, false));

            // 執行按鍵處理
            _normalMode.WaitForInput();

            // 驗證游標是否移動到行尾
            Assert.Equal(expectedX, _editor.CursorX);
            Assert.Equal(1, _editor.CursorY); // Y 應該不變
        }

        [Fact]
        public void TestSetViewPort()
        {
            // 設置新的 ViewPort
            int newX = 5;
            int newY = 2;
            int newWidth = 60;
            int newHeight = 20;
            
            _editor.SetViewPort(newX, newY, newWidth, newHeight);
            
            // 驗證 ViewPort 是否已正確設置
            Assert.Equal(newX, _editor.ViewPort.X);
            Assert.Equal(newY, _editor.ViewPort.Y);
            Assert.Equal(newWidth, _editor.ViewPort.Width);
            Assert.Equal(newHeight, _editor.ViewPort.Height);
        }

        [Fact]
        public void TestMoveCursorToEndOfLineWithChineseChars()
        {
            // 準備包含中文字符的文本
            // 由於中文字符在控制台中佔據雙寬度，這是測試游標定位的重要案例
            _editor.Texts.Clear();
            _editor.Texts.Add(new ConsoleText("Hello 閃電"));
            
            // 設置初始游標位置（非行尾）
            _editor.CursorX = 2; // 在 'l' 字符上
            _editor.CursorY = 0;
            
            // 模擬按下 '$' 鍵 (Shift+4, 對應 ConsoleKey.D4)
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('$', ConsoleKey.D4, true, false, false));
            
            // 執行按鍵處理
            _normalMode.WaitForInput();
            
            // 獲取第一行的長度以進行驗證
            int expectedX = _editor.Texts[0].Width - 2;
            
            // 驗證游標是否移動到行尾 ('電' 字符上)
            // 確保 VimEditor 在處理中文混合文本時能正確計算字符位置
            Assert.Equal(expectedX, _editor.CursorX);
            Assert.Equal(0, _editor.CursorY);
            
            // 注意：在 ConsoleText 的實現中，Width 屬性返回 Chars.Length
            // 即字符數量，而不考慮字符在顯示時的寬度
            // 對於中文字符而言，這確保了游標能正確定位到最後一個字符
        }
    }
} 