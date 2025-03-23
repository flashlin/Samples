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
    private int _endOffsetX;
    private int _endOffsetY;
    private int _endTextX;
    private int _endTextY;

    // 記錄選取的起始位置
    private int _startOffsetX;
    private int _startOffsetY;
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
        outputBuffer.Append($"\x1b[{Instance.Context.CursorY+1};{Instance.Context.CursorX+1}H");
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
        _startOffsetX = Instance.Context.OffsetX;
        _startOffsetY = Instance.Context.OffsetY;
        _startTextX = Instance.Context.CursorX;
        _startTextY = Instance.Context.CursorY;
        SaveLastPosition();
    }

    /// <summary>
    /// 檢查並調整游標位置和偏移量，確保游標在可見區域內
    /// </summary>
    private void AdjustCursorAndOffset()
    {
        // 調用 VimEditor 中的 AdjustCursorAndOffset 方法
        Instance.AdjustCursorPositionAndOffset(_endOffsetX, _endOffsetY);
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
            { new ConsoleKeyPattern(ConsoleKey.Escape), SwitchToVisualMode }
        });
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
        _endOffsetX = Instance.Context.OffsetX;
        _endOffsetY = Instance.Context.OffsetY;
        _endTextX = Instance.Context.CursorX;
        _endTextY = Instance.Context.CursorY;
    }

    /// <summary>
    /// 向上移動游標
    /// </summary>
    private void MoveCursorUp(List<ConsoleKeyInfo> keys)
    {
        _normalMode.MoveCursorUp(keys);
        SaveLastPosition();
    }

    private void SaveLastPosition()
    {
        _endOffsetX = Instance.Context.OffsetX;
        _endOffsetY = Instance.Context.OffsetY;
        _endTextX = Instance.Context.CursorX;
        _endTextY = Instance.Context.CursorY;
    }

    /// <summary>
    /// 切換到視覺模式
    /// </summary>
    private void SwitchToVisualMode(List<ConsoleKeyInfo> keys)
    {
        Instance.Mode = new VimNormalMode(Instance);
    }
} 