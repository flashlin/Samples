using Xunit;
using NSubstitute;
using System;
using TextCopy;

namespace VimSharpLib.Tests;

public class VimVisualModeTests
{
    private IConsoleDevice _mockConsole;
    private VimEditor _editor;

    public VimVisualModeTests()
    {
        _mockConsole = Substitute.For<IConsoleDevice>();
        _mockConsole.WindowWidth.Returns(80);
        _mockConsole.WindowHeight.Returns(25);
        _editor = new VimEditor(_mockConsole);
        _editor.Context.IsLineNumberVisible = false;
        _editor.Context.IsStatusBarVisible = false;
    }

    [Fact]
    public void Test1()
    {
        // Arrange
        // 設置 ViewPort = 1, 1, 40, 5
        _editor.Context.SetViewPort(1, 1, 40, 5);

        // 加載文本 "Hello World"
        _editor.OpenText("Hello World");

        // 確保編輯器處於正常模式
        _editor.Mode = new VimNormalMode(_editor);

        // Act
        // 按下 w 按鍵
        SetReadKey(ConsoleKeyPress.w);

        // 按下 v 按鍵
        SetReadKey(ConsoleKeyPress.v);

        // 按下 $ 按鍵
        SetReadKey(ConsoleKeyPress.DollarSign);

        // 按下 y 按鍵
        SetReadKey(ConsoleKeyPress.y);

        // Assert
        // 驗證游標位置
        Assert.Equal(11, _editor.Context.CursorX); // "Hello World" 的長度
        Assert.Equal(1, _editor.Context.CursorY); // 應該在第一行

        // 驗證當前模式是視覺模式
        Assert.IsType<VimNormalMode>(_editor.Mode);

        // 驗證剪貼簿內容
        Assert.Equal("World", ClipboardService.GetText());
    }

    private void SetReadKey(ConsoleKeyInfo keyInfo)
    {
        _mockConsole.ReadKey(Arg.Any<bool>()).Returns(keyInfo);
        _editor.WaitForInput();
    }
}