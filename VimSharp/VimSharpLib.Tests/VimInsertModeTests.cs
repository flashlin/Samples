using Xunit;
using NSubstitute;
using System;

namespace VimSharpLib.Tests
{
    public class VimInsertModeTests
    {
        private readonly IConsoleDevice _mockConsole;
        private readonly VimEditor _editor;

        public VimInsertModeTests()
        {
            _mockConsole = Substitute.For<IConsoleDevice>();
            _mockConsole.WindowWidth.Returns(80);
            _mockConsole.WindowHeight.Returns(25);
            _editor = new VimEditor(_mockConsole)
            {
                Context =
                {
                    IsLineNumberVisible = false,
                    IsStatusBarVisible = false
                }
            };
        }

        [Fact]
        public void TestInsertModeEnterKey()
        {
            // Arrange
            _editor.Context.SetViewPort(1, 1, 40, 5);
            _editor.OpenText("Hello World");
            
            // 移動游標到 'W' 的位置
            _editor.Context.CursorX = 7;
            
            // 進入插入模式
            _editor.Mode.PressKey(ConsoleKeyPress.i);
            
            // 按下 Enter 鍵
            _editor.Mode.PressKey(ConsoleKeyPress.Enter);
            
            // Assert
            Assert.Equal("World", _editor.GetCurrentLine().ToString());
            Assert.Equal(1, _editor.Context.CursorX);
            Assert.Equal(2, _editor.Context.CursorY);
            Assert.True(_editor.Mode is VimInsertMode, "_editor.Mode should be VimInsertMode");
        }
    }
} 