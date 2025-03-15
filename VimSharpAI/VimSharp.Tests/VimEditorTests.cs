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
    }
} 