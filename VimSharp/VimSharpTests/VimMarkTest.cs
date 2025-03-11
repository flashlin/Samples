using NUnit.Framework;
using FluentAssertions;
using NSubstitute;
using VimSharpLib;
using System;
using System.Linq;
using System.Reflection;

namespace VimSharpTests
{
    [TestFixture]
    public class VimMarkTest
    {
        private VimEditor _editor;
        private IConsoleDevice _mockConsole;

        [SetUp]
        public void Setup()
        {
            // 創建 Mock 的 IConsoleDevice
            _mockConsole = Substitute.For<IConsoleDevice>();
            _mockConsole.WindowWidth.Returns(80);
            _mockConsole.WindowHeight.Returns(25);
            
            // 創建使用 Mock 控制台的 VimEditor
            _editor = new VimEditor(_mockConsole);
        }

        [Test]
        public void WhenInVisualMode_PressV_SelectText_PressYKey_ShouldCopyToClipboard()
        {
            // 初始化 VimEditor
            _editor.Context.Texts.Clear();
            _editor.Context.Texts.Add(new ConsoleText());
            _editor.Context.Texts.Add(new ConsoleText());
            _editor.Context.Texts[0].SetText(0, "Hello, World!");
            _editor.Context.Texts[1].SetText(0, "Example.");
            
            // 設置初始游標位置
            _editor.Context.CursorX = 0;
            _editor.Context.CursorY = 0;
            
            // 按下右鍵按鈕7次，移動到 "W" 的位置
            for (int i = 0; i < 7; i++)
            {
                _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('\0', ConsoleKey.RightArrow, false, false, false));
                _editor.WaitForInput();
            }
            
            // 按下V按鈕一次，進入標記模式
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('v', ConsoleKey.V, false, false, false));
            _editor.WaitForInput();
            
            // 按下右鍵按鈕5次，移動到 "!" 的位置
            for (int i = 0; i < 5; i++)
            {
                _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('\0', ConsoleKey.RightArrow, false, false, false));
                _editor.WaitForInput();
            }
            
            // 按下 y 按鍵一次
            _mockConsole.ReadKey(true).Returns(new ConsoleKeyInfo('y', ConsoleKey.Y, false, false, false));
            _editor.WaitForInput();
            
            // 驗證 ClipboardBuffers 的內容
            _editor.ClipboardBuffers.Should().HaveCount(1);
            
            // 驗證剪貼簿中的內容是 "World!"
            string clipboardContent = new string(_editor.ClipboardBuffers[0].Chars.Select(c => c.Char).ToArray());
            clipboardContent.Should().Be("World!");
        }
    }
} 