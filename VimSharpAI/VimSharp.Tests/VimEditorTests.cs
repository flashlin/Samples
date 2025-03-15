using System;
using Xunit;
using NSubstitute;
using VimSharpLib;
using System.Collections.Generic;
using System.Linq;

namespace VimSharp.Tests
{
    public class VimEditorTests
    {
        private IConsoleDevice _mockConsole;
        private VimEditor _editor;

        public VimEditorTests()
        {
            // 設置模擬的控制台裝置
            _mockConsole = Substitute.For<IConsoleDevice>();
            _mockConsole.WindowWidth.Returns(80);
            _mockConsole.WindowHeight.Returns(25);

            // 創建 VimEditor 實例
            _editor = new VimEditor(_mockConsole);
        }

        [Fact]
        public void TestOpenTextWithChineseCharacters()
        {
            // 使用 OpenText 方法打開含有中文字符的文本
            _editor.OpenText("Hello 閃電");
            
            // 驗證第一行的長度
            // 根據 @rules.mdc，對於中文字符，每個字符後面應該有一個 '\0'
            // "Hello 閃電" 應該為 "H", "e", "l", "l", "o", " ", "閃", "\0", "電", "\0"
            // 總長度應該是 10 (6個英文字符 + 2個中文字符加上它們的 '\0')
            int expectedLength = 10;
            Assert.Equal(expectedLength, _editor.Texts[0].Width);
            
            // 驗證字符序列
            // 前6個字符應該是 "Hello "
            Assert.Equal('H', _editor.Texts[0].Chars[0].Char);
            Assert.Equal('e', _editor.Texts[0].Chars[1].Char);
            Assert.Equal('l', _editor.Texts[0].Chars[2].Char);
            Assert.Equal('l', _editor.Texts[0].Chars[3].Char);
            Assert.Equal('o', _editor.Texts[0].Chars[4].Char);
            Assert.Equal(' ', _editor.Texts[0].Chars[5].Char);
            
            // 第7個字符應該是 "閃"
            Assert.Equal('閃', _editor.Texts[0].Chars[6].Char);
            // 第8個字符應該是 '\0'
            Assert.Equal('\0', _editor.Texts[0].Chars[7].Char);
            // 第9個字符應該是 "電"
            Assert.Equal('電', _editor.Texts[0].Chars[8].Char);
            // 第10個字符應該是 '\0'
            Assert.Equal('\0', _editor.Texts[0].Chars[9].Char);
        }
        
        [Fact]
        public void TestCursorMovementAndViewPortOffset()
        {
            // 設置測試參數
            _editor.IsStatusBarVisible = true;
            _editor.SetViewPort(0, 0, 10, 4);
            
            // 創建五行測試文本
            string[] fiveLines = {
                "第一行",
                "第二行",
                "第三行",
                "第四行",
                "第五行"
            };
            
            // 使用 OpenText 方法載入文本
            _editor.OpenText(string.Join("\n", fiveLines));
            
            // 初始化 VimNormalMode
            var normalMode = new VimNormalMode { Instance = _editor };
            _editor.Mode = normalMode;
            
            // 驗證初始狀態
            Assert.Equal(0, _editor.CursorY);
            Assert.Equal(0, _editor.OffsetY);
            
            // 模擬按下 J 鍵四次
            for (int i = 0; i < 4; i++)
            {
                _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('j', ConsoleKey.J, false, false, false));
                normalMode.WaitForInput();
            }
            
            // 驗證游標位置和視口偏移
            Assert.Equal(3, _editor.CursorY); // 因為 ViewPort Height 規定只能顯示四行, 又有 StatusBar 用掉一行, 所以游標應該在第三行
            Assert.Equal(2, _editor.OffsetY); // 視口偏移應該是 2
            
            // 額外驗證 ViewPort 大小設置是否正確
            Assert.Equal(0, _editor.ViewPort.X);
            Assert.Equal(0, _editor.ViewPort.Y);
            Assert.Equal(10, _editor.ViewPort.Width);
            Assert.Equal(4, _editor.ViewPort.Height);
        }
    }
} 