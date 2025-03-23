namespace VimSharpLib;

using System.Text;
using System.Linq;
using System;
using System.Collections.Generic;

public class VimVisualMode : IVimMode
{
    private readonly KeyHandler _keyHandler;
    private readonly VimNormalMode _normalMode;

    // 記錄選取的結束位置
    private int _endTextX;
    private int _endTextY;

    // 記錄選取的起始位置
    private int _startTextX;
    private int _startTextY;

    public VimVisualMode(VimEditor instance)
    {
        Instance = instance;
        _normalMode = new VimNormalMode(instance);
        _keyHandler = new KeyHandler(instance.Console);
        InitializeKeyPatterns();
    }

    public VimEditor Instance { get; set; }

    public void PressKey(ConsoleKeyInfo keyInfo)
    {
        _keyHandler.PressKey(keyInfo);
    }

    public void AfterRender(StringBuilder outputBuffer)
    {
        // 設置控制台游標位置
        outputBuffer.Append($"\x1b[{Instance.Context.CursorY + 1};{Instance.Context.CursorX + 1}H");
        // 顯示游標
        outputBuffer.Append("\x1b[?25h");
        // 顯示方塊游標
        outputBuffer.Append("\x1b[2 q");
    }

    public void WaitForInput()
    {
        // 設置為方塊游標 (DECSCUSR 2)
        // Instance.Console.SetBlockCursor();
        _keyHandler.WaitForInput();
    }

    /// <summary>
    /// 設置選取的起始位置
    /// </summary>
    public void SetStartPosition()
    {
        _startTextX = Instance.GetActualTextX();
        _startTextY = Instance.GetActualTextY();
        SaveLastPosition();
    }

    /// <summary>
    /// 複製選取的文本
    /// </summary>
    private void CopySelectedText(List<ConsoleKeyInfo> keys)
    {
    }

    /// <summary>
    /// 高亮顯示選取的文本
    /// </summary>
    private void HighlightSelectedText()
    {
        var startOffset = Instance.Context.ComputeOffset(_startTextX, _startTextY);
        var endOffset = Instance.Context.ComputeOffset(_endTextX, _endTextY);
        Instance.Context.GetText(startOffset, endOffset - startOffset);
    }

    private void InitializeKeyPatterns()
    {
        _keyHandler.InitializeKeyPatterns(new Dictionary<IKeyPattern, Action<List<ConsoleKeyInfo>>>
        {
            { new ConsoleKeyPattern(ConsoleKey.LeftArrow), MoveCursorLeft },
            { new ConsoleKeyPattern(ConsoleKey.RightArrow), MoveCursorRight },
            { new ConsoleKeyPattern(ConsoleKey.UpArrow), MoveCursorUp },
            { new ConsoleKeyPattern(ConsoleKey.DownArrow), MoveCursorDown },
            { new ConsoleKeyPattern(ConsoleKey.Y), CopySelectedText },
            { new ConsoleKeyPattern(ConsoleKey.Escape), SwitchToVisualMode },
            { new CharKeyPattern('$'), MoveCursorToEndOfLine },
        });
    }

    private void MoveCursorToEndOfLine(List<ConsoleKeyInfo> keys)
    {
        _normalMode.MoveCursorToEndOfLine(keys);
        SaveLastPosition();
    }

    /// <summary>
    /// 向下移動游標
    /// </summary>
    private void MoveCursorDown(List<ConsoleKeyInfo> keys)
    {
        _normalMode.MoveCursorDown(keys);
        SaveLastPosition();
    }

    /// <summary>
    /// 向左移動游標
    /// </summary>
    private void MoveCursorLeft(List<ConsoleKeyInfo> keys)
    {
        _normalMode.MoveCursorLeft(keys);
        SaveLastPosition();
    }

    /// <summary>
    /// 向右移動游標
    /// </summary>
    private void MoveCursorRight(List<ConsoleKeyInfo> keys)
    {
        _normalMode.MoveCursorRight(keys);
        SaveLastPosition();
    }

    /// <summary>
    /// 向上移動游標
    /// </summary>
    private void MoveCursorUp(List<ConsoleKeyInfo> keys)
    {
        _normalMode.MoveCursorUp(keys);
        SaveLastPosition();
    }

    public void Render(ColoredChar[,] screenBuffer)
    {
        var startX = Instance.Context.ViewPort.X + Instance.Context.GetLineNumberWidth();
        var endX = Instance.Context.ViewPort.X + Instance.Context.ViewPort.Width - 1 -
                   Instance.Context.GetLineNumberWidth();
        var startY = Instance.Context.ViewPort.Y;
        var endY = Instance.Context.ViewPort.Y + Instance.Context.ViewPort.Height - 1 -
                   Instance.Context.StatusBarHeight;

        var startOffset = Instance.Context.ComputeOffset(_startTextX, _startTextY);
        var endOffset = Instance.Context.ComputeOffset(_endTextX, _endTextY);

        for (var y = startY; y <= endY; y++)
        {
            for (var x = startX; x <= endX; x++)
            {
                var viewTextX = Instance.Context.ComputeTextX(x);
                var viewTextY = Instance.Context.ComputeTextY(y);
                var viewOffset = Instance.Context.ComputeOffset(viewTextX, viewTextY);
                if (startOffset <= viewOffset && viewOffset <= endOffset)
                {
                    var character = screenBuffer[y, x];
                    if (!character.IsEmpty)
                    {
                        if (character.Char == ' ')
                        {
                            var newChar = new ColoredChar(' ', ConsoleColor.White, ConsoleColor.White);
                            screenBuffer[y, x] = newChar;
                        }
                        else
                        {
                            var newChar = new ColoredChar(character, ConsoleColor.Black, ConsoleColor.White);
                            screenBuffer[y, x] = newChar;
                        }
                    }
                }
            }
        }
    }

    private void SaveLastPosition()
    {
        _endTextX = Instance.GetActualTextX();
        _endTextY = Instance.GetActualTextY();
    }

    /// <summary>
    /// 切換到視覺模式
    /// </summary>
    private void SwitchToVisualMode(List<ConsoleKeyInfo> keys)
    {
        Instance.Mode = new VimNormalMode(Instance);
    }
}