using Xunit;
using NSubstitute;
using System;
using TextCopy;

namespace VimSharpLib.Tests;

public class VimVisualModeTests
{
    private readonly VimSharpTester _vimSharpTester = new();
    private readonly IConsoleDevice _mockConsole;
    private readonly VimEditor _editor;

    public VimVisualModeTests()
    {
        _editor = _vimSharpTester.CreateVimEditor();
        _mockConsole = _vimSharpTester.MockConsole;
        _editor.Context.IsLineNumberVisible = false;
        _editor.Context.IsStatusBarVisible = false;
    }

    [Fact]
    public void ShouldCopyWordToEndOfLineInVisualMode()
    {
        // Arrange
        // 設置 ViewPort = 1, 1, 40, 5
        _editor.Context.SetViewPort(1, 1, 40, 5);

        // 加載文本 "Hello World"
        _editor.OpenText("Hello World");

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

    [Fact]
    public void ShouldCopyAcrossLinesInVisualMode()
    {
        // Arrange
        // 設置 ViewPort = 1, 1, 40, 5
        _editor.Context.SetViewPort(1, 1, 40, 5);

        // 加載文本 "Hello World\nVim is a Editor"
        _editor.OpenText("Hello World\nVim is a Editor");
        
        // Act
        // 按下 w 按鍵
        SetReadKey(ConsoleKeyPress.w);
        
        // 按下 v 按鍵
        SetReadKey(ConsoleKeyPress.v);

        // 按下向下按鍵
        SetReadKey(ConsoleKeyPress.DownArrow);

        // 按下 ^ 按鍵
        SetReadKey(ConsoleKeyPress.Caret);
        Assert.Equal(1, _editor.Context.CursorX); // "Vim" 的 "v" 位置
        Assert.Equal(2, _editor.Context.CursorY); // 應該在第二行

        // 按下向右按鍵兩次
        SetReadKey(ConsoleKeyPress.RightArrow);
        SetReadKey(ConsoleKeyPress.RightArrow);

        // 按下 y 按鍵
        SetReadKey(ConsoleKeyPress.y);

        // Assert
        // 驗證游標位置
        Assert.Equal(3, _editor.Context.CursorX); // "Vim" 的 "m" 位置
        Assert.Equal(2, _editor.Context.CursorY); // 應該在第二行

        // 驗證當前模式是正常模式
        Assert.IsType<VimNormalMode>(_editor.Mode);

        // 驗證剪貼簿內容
        Assert.Equal("World\nVim", ClipboardService.GetText());
    }

    private void SetReadKey(ConsoleKeyInfo keyInfo)
    {
        _mockConsole.ReadKey(Arg.Any<bool>()).Returns(keyInfo);
        _editor.WaitForInput();
    }
}