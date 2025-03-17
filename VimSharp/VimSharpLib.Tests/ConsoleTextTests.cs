using Xunit;

namespace VimSharpLib.Tests
{
    public class ConsoleTextTests
    {
        [Fact]
        public void TestSetWidth()
        {
            // 創建新的 ConsoleText 實例
            var consoleText = new ConsoleText();
            
            // 設置寬度為 10
            consoleText.SetWidth(10);
            Assert.Equal(10, consoleText.Width);
            
            // 設置寬度為 20 (擴展)
            consoleText.SetWidth(20);
            Assert.Equal(20, consoleText.Width);
            
            // 設置寬度為 5 (縮小)
            consoleText.SetWidth(5);
            Assert.Equal(5, consoleText.Width);
        }
        
        [Fact]
        public void TestSetText()
        {
            // 創建新的 ConsoleText 實例
            var consoleText = new ConsoleText();
            
            // 設置文本 "123"
            consoleText.SetText(0, "123");
            
            // 驗證寬度為 3
            Assert.Equal(3, consoleText.Width);
        }
        
        [Fact]
        public void TestSetTextWithChineseCharacters()
        {
            // 創建新的 ConsoleText 實例
            var consoleText = new ConsoleText();
            
            // 設置包含中文字符的文本 "1中2"
            consoleText.SetText(0, "1中2");
            
            // 驗證寬度為 4 (中文字符占用 2 個位置)
            Assert.Equal(4, consoleText.Width);
        }
        
        [Fact]
        public void TestCharWidthWithChineseCharacters()
        {
            // 測試單個字符的寬度計算
            Assert.Equal(1, '1'.GetCharWidth());
            Assert.Equal(2, '中'.GetCharWidth());
            Assert.Equal(1, '2'.GetCharWidth());
            
            // 測試字符串的總寬度計算
            Assert.Equal(4, "1中2".GetStringDisplayWidth());
        }
        
        [Fact]
        public void TestSetWidthBeforeSetText()
        {
            // 創建新的 ConsoleText 實例
            var consoleText = new ConsoleText();
            
            // 計算 "1中2" 的顯示寬度
            int width = "1中2".GetStringDisplayWidth();
            Assert.Equal(4, width);
            
            // 手動設置足夠的寬度
            consoleText.SetWidth(width);
            Assert.Equal(4, consoleText.Width);
            
            // 現在嘗試設置文本
            consoleText.SetText(0, "1中2");
            
            // 驗證寬度保持不變
            Assert.Equal(4, consoleText.Width);
            
            // 驗證字符是否正確設置
            Assert.Equal('1', consoleText.Chars[0].Char);
            Assert.Equal('中', consoleText.Chars[1].Char);
            Assert.Equal('\0', consoleText.Chars[2].Char); // 中文字符佔用兩個位置，第二個應該是特殊標記
            Assert.Equal('2', consoleText.Chars[3].Char);
        }
        
        [Fact]
        public void TestFindLastCharIndex()
        {
            // 創建新的 ConsoleText 實例
            var consoleText = new ConsoleText();
            
            // 設置文本 "Hello "
            consoleText.SetText(0, "Hello ");
            
            // 調用 FindLastCharIndex 方法
            var lastIndex = consoleText.FindLastCharIndex();
            
            // 驗證 lastIndex 等於 4（最後一個非空字符 'o' 的索引）
            Assert.Equal(4, lastIndex);
        }
        
        [Fact]
        public void TestFindLastCharIndex2()
        {
            // 創建新的 ConsoleText 實例
            var consoleText = new ConsoleText();
            
            // 設置文本 "Hello "
            consoleText.SetText(0, "Hello 閃電");
            
            // 調用 FindLastCharIndex 方法
            var lastIndex = consoleText.FindLastCharIndex();
            
            Assert.Equal(8, lastIndex);
        }
    }
} 