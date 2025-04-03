using Xunit;
using NSubstitute;

namespace VimSharpLib.Tests
{
    public class VimEditorRenderTests
    {
        private VimSharpTester _vimSharpTester = new();  
        private VimEditor _editor;

        public VimEditorRenderTests()
        {
            _editor = _vimSharpTester.CreateVimEditor();
            _editor.Context.IsLineNumberVisible = false;
            _editor.Context.IsStatusBarVisible = false;
            _editor.Context.SetViewPort(0, 0, 10, 5);
        }

        [Fact]
        public void TestChineseCharacterRendering()
        {
            // Arrange
            _editor.Context.IsLineNumberVisible = false; // 關閉行號顯示以簡化測試
            _editor.Context.IsStatusBarVisible = false; // 關閉狀態欄以簡化測試
            _editor.Context.ViewPort = new ViewArea(0, 0, 10, 5); // 設置 ViewPort 為 (0,0,10,5)
            _editor.Context.OffsetX = 0;
            _editor.Context.OffsetY = 0;
            
            // 設置測試文本，包含中文字符
            string text = "1中2";
            _editor.OpenText(text);
            
            // 創建自定義大小的 screenBuffer
            var screenBuffer = _vimSharpTester.ScreenBuffer;
            
            // Act
            _editor.Render(screenBuffer);
            
            // Assert
            // 檢查中文字符的渲染，中文字符應該佔用兩個位置（第二個位置為 '\0'）
            Assert.Equal('1', screenBuffer[0, 0].Char);
            Assert.Equal('中', screenBuffer[0, 1].Char);
            Assert.Equal('\0', screenBuffer[0, 2].Char);
            Assert.Equal('2', screenBuffer[0, 3].Char);
            
            // 另外檢查第一個中文字符的顏色
            Assert.Equal(ConsoleColor.White, screenBuffer[0, 1].ForegroundColor);
            Assert.Equal(ConsoleColor.Black, screenBuffer[0, 1].BackgroundColor);

            // 檢查 ViewPort 以外的內容, 不應該被改變. 也就是要求 Render 不會改變 ViewPort 以外的內容
            Assert.Equal('.', screenBuffer[0, 11].Char);
            Assert.Equal('.', screenBuffer[1, 11].Char);
            Assert.Equal('.', screenBuffer[2, 11].Char);
        }

        [Fact]
        public void TestChineseCharacterRenderingWithOffsetViewPort()
        {
            // Arrange
            _editor.Context.IsLineNumberVisible = false; // 關閉行號顯示以簡化測試
            _editor.Context.IsStatusBarVisible = false; // 關閉狀態欄以簡化測試
            _editor.Context.ViewPort = new ViewArea(1, 0, 10, 5); // 設置 ViewPort 為 (1,0,10,5)，X 座標從 1 開始
            _editor.Context.OffsetX = 0;
            _editor.Context.OffsetY = 0;
            
            // 設置測試文本，包含中文字符
            string text = "1中2";
            _editor.OpenText(text);
            
            // 創建自定義大小的 screenBuffer
            var screenBuffer = _vimSharpTester.ScreenBuffer;
            
            // Act
            _editor.Render(screenBuffer);
            
            // Assert
            // 檢查中文字符的渲染，中文字符應該佔用兩個位置（第二個位置為 '\0'）
            // 由於 ViewPort 的 X 從 1 開始，所以文本應該從 screenBuffer[0, 1] 開始渲染
            Assert.Equal('1', screenBuffer[0, 1].Char);
            Assert.Equal('中', screenBuffer[0, 2].Char);
            Assert.Equal('\0', screenBuffer[0, 3].Char);
            Assert.Equal('2', screenBuffer[0, 4].Char);
            
            // 另外檢查第一個中文字符的顏色
            Assert.Equal(ConsoleColor.White, screenBuffer[0, 2].ForegroundColor);
            Assert.Equal(ConsoleColor.Black, screenBuffer[0, 2].BackgroundColor);
        }

        [Fact]
        public void TestChineseCharacterRenderingWithLineNumbers()
        {
            // Arrange
            _editor.Context.IsLineNumberVisible = true; // 開啟行號顯示
            _editor.Context.IsStatusBarVisible = false; // 關閉狀態欄以簡化測試
            _editor.Context.ViewPort = new ViewArea(0, 0, 10, 5); // 設置 ViewPort 為 (0,0,10,5)
            _editor.Context.OffsetX = 0;
            _editor.Context.OffsetY = 0;
            
            // 設置測試文本，包含中文字符
            string text = "1中2";
            _editor.OpenText(text);
            
            // 創建自定義大小的 screenBuffer
            var screenBuffer = _vimSharpTester.ScreenBuffer;
            
            // Act
            _editor.Render(screenBuffer);
            
            // Assert
            // 行號寬度為 1 + 1 = 2 (1位數字 + 1位空格)
            int lineNumberWidth = 2;
            
            // 檢查行號是否正確渲染
            Assert.Equal('1', screenBuffer[0, 0].Char);
            Assert.Equal(' ', screenBuffer[0, 1].Char);
            
            // 檢查中文字符的渲染，中文字符應該佔用兩個位置，並且應該在行號之後
            Assert.Equal('1', screenBuffer[0, lineNumberWidth].Char);
            Assert.Equal('中', screenBuffer[0, lineNumberWidth + 1].Char);
            Assert.Equal('\0', screenBuffer[0, lineNumberWidth + 2].Char);
            Assert.Equal('2', screenBuffer[0, lineNumberWidth + 3].Char);
            
            // 檢查行號顏色
            Assert.Equal(ConsoleColor.Yellow, screenBuffer[0, 0].ForegroundColor);
            Assert.Equal(ConsoleColor.DarkBlue, screenBuffer[0, 0].BackgroundColor);
            
            // 檢查中文字符的顏色
            Assert.Equal(ConsoleColor.White, screenBuffer[0, lineNumberWidth + 1].ForegroundColor);
            Assert.Equal(ConsoleColor.Black, screenBuffer[0, lineNumberWidth + 1].BackgroundColor);
        }
    }
} 