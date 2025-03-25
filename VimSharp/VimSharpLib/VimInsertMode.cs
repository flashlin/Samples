namespace VimSharpLib;
using System.Text;
using System.Linq;
using System;
using System.Collections.Generic;

public class VimInsertMode : IVimMode
{
    private readonly KeyHandler _keyHandler;

    public VimInsertMode(VimEditor instance)
    {
        Instance = instance;
        _keyHandler = new KeyHandler(instance.Console);
        InitializeKeyHandler();
    }
    
    public VimEditor Instance { get; }
    
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
        // 設置為垂直線游標 (DECSCUSR 6)
        outputBuffer.Append("\x1b[6 q");
    }

    public void Render(ColoredChar[,] screenBuffer)
    {
    }

    /// <summary>
    /// 初始化按鍵處理邏輯
    /// </summary>
    private void InitializeKeyHandler()
    {
        _keyHandler.InitializeKeyPatterns(new Dictionary<IKeyPattern, Action<List<ConsoleKeyInfo>>>
        {
            // 註冊基本功能鍵
            { new ConsoleKeyPattern(ConsoleKey.Escape), HandleEscape },
            { new ConsoleKeyPattern(ConsoleKey.Backspace), HandleBackspace },
            { new ConsoleKeyPattern(ConsoleKey.Delete), HandleDeleteKey },
            { new ConsoleKeyPattern(ConsoleKey.LeftArrow), MoveCursorLeft },
            { new ConsoleKeyPattern(ConsoleKey.RightArrow), MoveCursorRight },
            { new ConsoleKeyPattern(ConsoleKey.UpArrow), MoveCursorUp },
            { new ConsoleKeyPattern(ConsoleKey.DownArrow), MoveCursorDown },
            { new ConsoleKeyPattern(ConsoleKey.Enter), HandleEnterKey },
            { new AnyKeyPattern(), HandleAnyKeyInput }
        });
    }

    /// <summary>
    /// 處理任意鍵輸入，會調用 HandleCharInput 方法
    /// </summary>
    private void HandleAnyKeyInput(List<ConsoleKeyInfo> keys)
    {
        if (keys.Count > 0)
        {
            var keyInfo = keys[0];
            var key = keyInfo.Key;
            char keyChar;
            
            // 處理數字鍵
            if (key >= ConsoleKey.D0 && key <= ConsoleKey.D9)
            {
                keyChar = (char)('0' + (key - ConsoleKey.D0));
                HandleCharInput(keyChar);
            }
            // 處理字母鍵
            else if (key >= ConsoleKey.A && key <= ConsoleKey.Z)
            {
                keyChar = (char)('a' + (key - ConsoleKey.A));
                HandleCharInput(keyChar);
            }
            else
            {
                // 對於其他按鍵，使用默認轉換
                keyChar = key.ToChar();
                if (keyChar != '\0')
                {
                    HandleCharInput(keyChar);
                }
            }
            
            // 清空按鍵緩衝區以準備下一次輸入
            _keyHandler.Clear();
        }
    }

    /// <summary>
    /// 檢查並調整游標位置和偏移量，確保游標在可見區域內
    /// </summary>
    private void AdjustCursorAndOffset()
    {
        // 調用 VimEditor 中的 AdjustCursorAndOffset 方法
        Instance.AdjustCursorPositionAndOffset(Instance.Context.CursorX, Instance.Context.CursorY);
    }
    
    /// <summary>
    /// 切換到普通模式
    /// </summary>
    private void HandleEscape(List<ConsoleKeyInfo> keys)
    {
        // 獲取當前行
        var currentLine = Instance.GetCurrentLine();
        var textX = Instance.GetActualTextX();
        var isEndOfLine = textX >= currentLine.Chars.Length;
        
        // 記錄當前游標位置和偏移量
        int cursorX = Instance.Context.CursorX;
        int offsetX = Instance.Context.OffsetX;
        
        // 切換到普通模式
        Instance.Mode = new VimNormalMode(Instance);
        
        // 如果游標在行尾，則處理游標位置
        if (isEndOfLine)
        {
            // 1. 減少 GetActualX()，即 textX - 1
            // 2. 減少 CursorX
            Instance.Context.CursorX = cursorX - 1;
            
            // 3. 檢查 CursorX 是否小於 ViewPort.X - GetLineNumberWidth()
            int minX = Instance.Context.ViewPort.X - Instance.Context.GetLineNumberWidth();
            if (Instance.Context.CursorX < minX)
            {
                // 設置 CursorX 為最小允許值
                Instance.Context.CursorX = minX;
                
                // 減少 OffsetX
                Instance.Context.OffsetX = offsetX - 1;
                
                // 確保 OffsetX 不小於 0
                if (Instance.Context.OffsetX < 0)
                {
                    Instance.Context.OffsetX = 0;
                }
            }
        }
    }

    /// <summary>
    /// 處理退格鍵
    /// </summary>
    private void HandleBackspace(List<ConsoleKeyInfo> keys)
    {
        // 正常環境的處理邏輯
        if (Instance.Context.CursorX > 0)
        {
            // 獲取當前行
            var index = Instance.GetActualTextY();
            
            // 確保索引在有效範圍內
            if (index < 0 || index >= Instance.Context.Texts.Count)
                return;
                
            var currentLine = Instance.Context.Texts[index];
            if (currentLine == null)
                return;

            // 獲取當前文本
            string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
            
            // 檢查文本是否為空
            if (string.IsNullOrEmpty(currentText))
                return;

            // 計算實際索引位置（考慮偏移量）
            int actualTextX = Instance.GetActualTextX();
            
            // 確保索引在有效範圍內
            if (actualTextX <= 0 || actualTextX > currentText.Length)
                return;

            // 獲取要刪除的字符
            char charToDelete = currentText[actualTextX - 1];

            // 刪除字符
            string newText = currentText.Remove(actualTextX - 1, 1);

            // 更新文本
            currentLine.SetText(0, newText);

            // 移動光標（考慮中文字符寬度）
            Instance.Context.CursorX -= charToDelete.GetCharWidth();

            // 清除屏幕並重新渲染整行（對於 Backspace，我們需要重新渲染整行）
            Instance.Render();
        }
    }
    
    /// <summary>
    /// 向左移動游標
    /// </summary>
    private void MoveCursorLeft(List<ConsoleKeyInfo> keys)
    {
        // 檢查是否到達左邊界
        var minX = Instance.Context.ViewPort.X + Instance.Context.GetLineNumberWidth();
        if (Instance.Context.CursorX <= minX)
        {
            // 如果有水平偏移，則減少偏移而不是移動游標
            if (Instance.Context.OffsetX > 0)
            {
                Instance.Context.OffsetX--;
                return;
            }
            return; // 已經到達最左邊，不能再移動
        }
        // 正常情況下向左移動游標
        Instance.Context.CursorX--;
    }
    
    /// <summary>
    /// 向右移動游標
    /// </summary>
    private void MoveCursorRight(List<ConsoleKeyInfo> keys)
    {
        // 獲取當前行
        var currentLine = Instance.GetCurrentLine();
        // 檢查是否到達右邊界
        if (Instance.Context.CursorX >= Instance.Context.ViewPort.Right)
        {
            // 如果還有更多文本可以顯示，則增加水平偏移
            if (Instance.Context.OffsetX < currentLine.Width - Instance.Context.ViewPort.Width)
            {
                Instance.Context.OffsetX++;
                CheckCursorX();
                return;
            }
            CheckCursorX();
            return; // 已經到達最右邊，不能再移動
        }
        // 正常情況下向右移動游標
        Instance.Context.CursorX++;
        CheckCursorX();
    }
    
    /// <summary>
    /// 向上移動游標
    /// </summary>
    private void MoveCursorUp(List<ConsoleKeyInfo> keys)
    {
        if (Instance.Context.CursorY <= Instance.Context.ViewPort.Y)
        {
            if (Instance.Context.OffsetY > 0)
            {
                Instance.Context.OffsetY--;
                CheckCursorX();
                return;
            }
            CheckCursorX();
            return;
        }
        Instance.Context.CursorY--;
        CheckCursorX();
    }
    
    /// <summary>
    /// 向下移動游標
    /// </summary>
    private void MoveCursorDown(List<ConsoleKeyInfo> keys)
    {
        if (Instance.Context.CursorY >= Instance.Context.ViewPort.Bottom - Instance.Context.StatusBarHeight)
        {
            if (Instance.Context.OffsetY < Instance.Context.Texts.Count - Instance.Context.ViewPort.Height)
            {
                Instance.Context.OffsetY++;
                CheckCursorX();
                return;
            }
            CheckCursorX();
            return;
        }
        Instance.Context.CursorY++;
        CheckCursorX();
    }

    private void CheckCursorX()
    {
        var textX = Instance.GetActualTextX();
        var currentLine = Instance.GetCurrentLine();
        if (textX >= currentLine.Width)
        {
            new VimNormalMode(Instance).MoveCursorToEndOfLine([]);
            Instance.Context.CursorX += 1;
        }
    }

    /// <summary>
    /// 處理 Enter 鍵
    /// </summary>
    private void HandleEnterKey(List<ConsoleKeyInfo> keys)
    {
        // 獲取當前行
        var enterCurrentLine = Instance.GetCurrentLine();
        var enterActualIndex = Instance.GetActualTextX();
        var enterActualY = Instance.GetActualTextY();
        
        // 檢查游標後面是否有內容
        string remainingText = "";
        if (enterActualIndex < enterCurrentLine.Width)
        {
            // 獲取游標後面的內容
            remainingText = enterCurrentLine.GetText(enterActualIndex);
            
            // 修改當前行，只保留游標前面的內容
            string newCurrentText = enterCurrentLine.Substring(0, enterActualIndex);
            enterCurrentLine.SetText(0, newCurrentText);
        }

        // 在當前行後插入新行
        Instance.Context.CursorX = Instance.Context.ViewPort.X + Instance.Context.GetLineNumberWidth();
        // 新增一行
        var newLine = new ConsoleText();
        Instance.Context.Texts.Insert(enterActualY+1, newLine);
        
        // 如果有剩餘內容，設置到新行
        if (!string.IsNullOrEmpty(remainingText))
        {
            newLine.SetText(0, remainingText);
        }
        
        MoveCursorDown([ConsoleKeyPress.DownArrow]);
    }
    
    /// <summary>
    /// 處理一般字符輸入
    /// </summary>
    private void HandleCharInput(char keyChar)
    {
        var currentLine = Instance.GetCurrentLine();
        var textX = Instance.GetActualTextX();
        currentLine.InsertText(textX, keyChar.ToString());
        MoveCursorRight([ConsoleKeyPress.RightArrow]);
    }

    /// <summary>
    /// 處理 Delete 鍵
    /// </summary>
    private void HandleDeleteKey(List<ConsoleKeyInfo> keys)
    {
        // 獲取當前行
        var currentLine = Instance.GetCurrentLine();
        
        // 獲取當前文本
        string currentText = new string(currentLine.Chars.Select(c => c.Char).Where(c => c != '\0').ToArray());
        
        // 計算實際索引位置
        int actualTextX = Instance.GetActualTextX();
        
        // 如果游標不在文本末尾，刪除游標後的字符
        if (!string.IsNullOrEmpty(currentText) && actualTextX >= 0 && actualTextX < currentText.Length)
        {
            string newText = currentText.Remove(actualTextX, 1);
            currentLine.SetText(0, newText);
            
            // 重新渲染
            Instance.Render();
        }
    }

    public void WaitForInput()
    {
        Instance.Console.SetCursorPosition(Instance.Context.CursorX, Instance.Context.CursorY);
        // 設置為垂直線游標 (DECSCUSR 6)
        Instance.Console.SetLineCursor();
        
        // 確保當前行存在
        if (Instance.Context.Texts.Count <= Instance.Context.CursorY)
        {
            Instance.Context.Texts.Add(new ConsoleText());
        }
        
        // 直接調用 KeyHandler 處理按鍵輸入
        _keyHandler.WaitForInput();
    }
}