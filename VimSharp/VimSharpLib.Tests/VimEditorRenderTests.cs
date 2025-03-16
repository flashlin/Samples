using Xunit;

namespace VimSharpLib.Tests
{
    public class VimEditorRenderTests
    {
        /// <summary>
        /// 創建自定義大小的 screenBuffer
        /// </summary>
        /// <param name="height">緩衝區高度</param>
        /// <param name="width">緩衝區寬度</param>
        /// <returns>初始化的 ColoredChar 陣列</returns>
        private ColoredChar[,] CreateScreenBuffer(int height = 25, int width = 80)
        {
            var screenBuffer = new ColoredChar[height, width];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    screenBuffer[y, x] = new ColoredChar(' ', ConsoleColor.White, ConsoleColor.Black);
                }
            }
            return screenBuffer;
        }

        [Fact]
        public void TestChineseCharacterRendering()
        {
            // Arrange
            var mockConsole = new MockConsoleDevice { WindowWidth = 80, WindowHeight = 25 };
            var editor = new VimEditor(mockConsole);
            editor.Context.IsLineNumberVisible = false; // 關閉行號顯示以簡化測試
            editor.Context.IsStatusBarVisible = false; // 關閉狀態欄以簡化測試
            editor.Context.ViewPort = new ViewArea(0, 0, 10, 5); // 設置 ViewPort 為 (0,0,10,5)
            editor.Context.OffsetX = 0;
            editor.Context.OffsetY = 0;
            
            // 設置測試文本，包含中文字符
            string text = "1中2";
            editor.OpenText(text);
            
            // 創建自定義大小的 screenBuffer
            var screenBuffer = CreateScreenBuffer();
            
            // Act
            editor.Render(screenBuffer);
            
            // Assert
            // 檢查中文字符的渲染，中文字符應該佔用兩個位置（第二個位置為 '\0'）
            Assert.Equal('1', screenBuffer[0, 0].Char);
            Assert.Equal('中', screenBuffer[0, 1].Char);
            Assert.Equal('\0', screenBuffer[0, 2].Char);
            Assert.Equal('2', screenBuffer[0, 3].Char);
            
            // 另外檢查第一個中文字符的顏色
            Assert.Equal(ConsoleColor.White, screenBuffer[0, 1].ForegroundColor);
            Assert.Equal(ConsoleColor.DarkGray, screenBuffer[0, 1].BackgroundColor);
        }

        [Fact]
        public void TestChineseCharacterRenderingWithOffsetViewPort()
        {
            // Arrange
            var mockConsole = new MockConsoleDevice { WindowWidth = 80, WindowHeight = 25 };
            var editor = new VimEditor(mockConsole);
            editor.Context.IsLineNumberVisible = false; // 關閉行號顯示以簡化測試
            editor.Context.IsStatusBarVisible = false; // 關閉狀態欄以簡化測試
            editor.Context.ViewPort = new ViewArea(1, 0, 10, 5); // 設置 ViewPort 為 (1,0,10,5)，X 座標從 1 開始
            editor.Context.OffsetX = 0;
            editor.Context.OffsetY = 0;
            
            // 設置測試文本，包含中文字符
            string text = "1中2";
            editor.OpenText(text);
            
            // 創建自定義大小的 screenBuffer
            var screenBuffer = CreateScreenBuffer();
            
            // Act
            editor.Render(screenBuffer);
            
            // Assert
            // 檢查中文字符的渲染，中文字符應該佔用兩個位置（第二個位置為 '\0'）
            // 由於 ViewPort 的 X 從 1 開始，所以文本應該從 screenBuffer[0, 1] 開始渲染
            Assert.Equal('1', screenBuffer[0, 1].Char);
            Assert.Equal('中', screenBuffer[0, 2].Char);
            Assert.Equal('\0', screenBuffer[0, 3].Char);
            Assert.Equal('2', screenBuffer[0, 4].Char);
            
            // 另外檢查第一個中文字符的顏色
            Assert.Equal(ConsoleColor.White, screenBuffer[0, 2].ForegroundColor);
            Assert.Equal(ConsoleColor.DarkGray, screenBuffer[0, 2].BackgroundColor);
        }
    }
} 