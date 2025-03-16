using Xunit;
using NSubstitute;
using System;

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
            var mockConsole = new MockConsoleDevice { WindowWidth = 80, WindowHeight = 25 };
            var editor = new VimEditor(mockConsole);
            
            // 設置 ViewPort = 0, 0, 10, 5
            editor.Context.SetViewPort(0, 0, 10, 5);
            
            // 加載文本 "Hello"
            editor.OpenText("Hello");
            
            // 確保編輯器處於正常模式
            editor.Mode = new VimNormalMode(editor);
            
            // Act
            // 按下向右按鍵 6 次
            for (int i = 0; i < 6; i++)
            {
                editor.Mode.PressKey(ConsoleKey.RightArrow);
            }
            
            // Assert
            // 驗證 CursorX 應該是 4
            Assert.Equal(4, editor.Context.CursorX);
        }

        [Fact]
        public void TestRightArrowKeyWithMultilineText()
        {
            // Arrange
            var mockConsole = new MockConsoleDevice { WindowWidth = 80, WindowHeight = 25 };
            var editor = new VimEditor(mockConsole);
            
            // 設置 ViewPort = 0, 0, 10, 5
            editor.Context.SetViewPort(0, 0, 10, 5);
            
            // 加載多行文本 "Hello\r\nFlash"
            editor.OpenText("Hello\r\nFlash");
            
            // 確保編輯器處於正常模式
            editor.Mode = new VimNormalMode(editor);
            
            // Act
            // 按下向右按鍵 6 次
            for (int i = 0; i < 6; i++)
            {
                editor.Mode.PressKey(ConsoleKey.RightArrow);
            }
            
            // Assert
            // 驗證 CursorX 應該是 4
            Assert.Equal(4, editor.Context.CursorX);
            // 驗證 CursorY 應該是 0（第一行）
            Assert.Equal(0, editor.Context.CursorY);
        }
        
        [Fact]
        public void TestRightArrowAndDownArrowKeys()
        {
            // Arrange
            var mockConsole = new MockConsoleDevice { WindowWidth = 80, WindowHeight = 25 };
            var editor = new VimEditor(mockConsole);
            
            // 設置 ViewPort = 0, 0, 10, 5
            editor.Context.SetViewPort(0, 0, 10, 5);
            
            // 加載多行文本 "Hello\r\nHi"
            editor.OpenText("Hello\r\nHi");
            
            // 確保編輯器處於正常模式
            editor.Mode = new VimNormalMode(editor);
            
            // Act
            // 按下向右按鍵 6 次
            for (int i = 0; i < 6; i++)
            {
                editor.Mode.PressKey(ConsoleKey.RightArrow);
            }
            
            // 按下向下按鍵 1 次
            editor.Mode.PressKey(ConsoleKey.DownArrow);
            
            // Assert
            // 驗證 CursorX 應該是 1
            Assert.Equal(1, editor.Context.CursorX);
            // 驗證 CursorY 應該是 1（第二行）
            Assert.Equal(1, editor.Context.CursorY);
        }
        
        [Fact]
        public void TestInitWithLineNumbers()
        {
            // Arrange
            var mockConsole = new MockConsoleDevice { WindowWidth = 80, WindowHeight = 25 };
            var editor = new VimEditor(mockConsole);
            
            // 設置 ViewPort = 0, 0, 10, 5
            editor.Context.SetViewPort(0, 0, 10, 5);
            
            // 啟用行號顯示
            editor.Context.IsLineNumberVisible = true;
            
            // 加載多行文本 "Hello\r\nHi"
            editor.OpenText("Hello\r\nHi");
            
            // Assert
            // 驗證初始游標位置
            // 由於有兩行文本，行號寬度為 1 位數字 + 1 位空格 = 2
            int expectedLineNumberWidth = 2;
            
            // 驗證 CursorX 應該是 ViewPort.X + 行號寬度
            Assert.Equal(editor.Context.ViewPort.X + expectedLineNumberWidth, editor.Context.CursorX);
            
            // 驗證 CursorY 應該是 ViewPort.Y
            Assert.Equal(editor.Context.ViewPort.Y, editor.Context.CursorY);

            // Act
            // 按下向右按鍵 6 次
            for (int i = 0; i < 6; i++)
            {
                editor.Mode.PressKey(ConsoleKey.RightArrow);
            }
            
            // 按下向下按鍵 1 次
            editor.Mode.PressKey(ConsoleKey.DownArrow);

            // Assert
             // 驗證 CursorX 應該是 3
            Assert.Equal(3, editor.Context.CursorX);
            // 驗證 CursorY 應該是 1（第二行）
            Assert.Equal(1, editor.Context.CursorY);
        }
        
        [Fact]
        public void TestRightArrowKeyWithChineseCharacter()
        {
            // Arrange
            var mockConsole = new MockConsoleDevice { WindowWidth = 80, WindowHeight = 25 };
            var editor = new VimEditor(mockConsole);
            
            // 設置 ViewPort = 0, 0, 10, 5
            editor.Context.SetViewPort(0, 0, 10, 5);
            
            // 加載包含中文字符的文本 "閃1"
            editor.OpenText("閃1");
            
            // 確保編輯器處於正常模式
            editor.Mode = new VimNormalMode(editor);
            
            // Act
            // 按下向右按鍵 1 次
            editor.Mode.PressKey(ConsoleKey.RightArrow);
            
            // Assert
            // 驗證 CursorX 應該是 2（因為中文字符"閃"佔用兩個字符寬度）
            Assert.Equal(2, editor.Context.CursorX);

            // 按下向左按鍵 1 次
            editor.Mode.PressKey(ConsoleKey.LeftArrow);

            // Assert
            // 驗證 CursorX 應該是 0
            Assert.Equal(0, editor.Context.CursorX);
        }

        [Fact]
        public void TestDollarKeyMovesToEndOfLine()
        {
            // Arrange
            var mockConsole = new MockConsoleDevice { WindowWidth = 80, WindowHeight = 25 };
            var editor = new VimEditor(mockConsole);
            
            // 設置 ViewPort = 0, 0, 10, 5
            editor.Context.SetViewPort(0, 0, 10, 5);
            
            // 加載文本 "Hello"
            editor.OpenText("Hello");
            
            // 確保編輯器處於正常模式
            editor.Mode = new VimNormalMode(editor);
            
            // Act
            // 按下 '$' 按鍵
            editor.Mode.PressKey(ConsoleKey.D4); // '$' 對應 Shift+4
            
            // Assert
            // 驗證 CursorX 應該是 4（"Hello" 的長度，游標位於最後一個字符上面）
            Assert.Equal(4, editor.Context.CursorX);
        }

        [Fact]
        public void TestDollarKeyAndWaitForInput()
        {
            // Arrange
            // 加載文本 "Hello"
            _editor.OpenText("Hello");
            
            // Act
            // 設置 ReadKey 返回 $ 按鍵 (Shift+4)
            SetReadKey('$');
            
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
            SetReadKey('$');
            
            // Assert
            // 驗證 CursorX 應該是 8
            // "Hi " 佔 3 個字符寬度，"閃" 佔 2 個字符寬度，"電" 佔 2 個字符寬度
            // 最後一個字, 原本是6, 但因為是中文字, 所以游標位置是 5
            Assert.Equal(5, _editor.Context.CursorX);
            
            // 確認模擬的 ReadKey 方法被調用了一次
            _mockConsole.Received(1).ReadKey(Arg.Any<bool>());
        }

        private void SetReadKey(char key)
        {
            switch(key)
            {
                case '$':
                    _mockConsole.ReadKey(Arg.Any<bool>()).Returns(
                        new ConsoleKeyInfo('$', ConsoleKey.D4, true, false, false)
                    );
                    break;
                default:
                    throw new ArgumentException($"Unsupported key: {key}");
            }
            // 調用 WaitForInput 方法，這將觸發模擬的 ReadKey 方法
            _editor.WaitForInput();
        }
    }
} 

