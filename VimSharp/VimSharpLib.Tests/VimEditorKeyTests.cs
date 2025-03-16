using Xunit;

namespace VimSharpLib.Tests
{
    public class VimEditorKeyTests
    {
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
    }
} 