namespace VimSharpLib;
using System.Text;
using System.Linq;
using System;
using System.Collections.Generic;

public class VimVisualMode : IVimMode
{
    private readonly KeyHandler _keyHandler;
    
    // 記錄選取的起始位置
    private int _startCursorX;
    private int _startCursorY;
    
    // 記錄選取的結束位置
    private int _endCursorX;
    private int _endCursorY;
    
    public VimVisualMode(VimEditor instance)
    {
        Instance = instance;
        _keyHandler = new KeyHandler(instance.Console);
        InitializeKeyPatterns();
    }
    public VimEditor Instance { get; set; }
    
    public void PressKey(ConsoleKey key)
    {
        _keyHandler.PressKey(key);
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
    /// 設置選取的起始位置
    /// </summary>
    public void SetStartPosition(int x, int y)
    {
        _startCursorX = x;
        _startCursorY = y;
        _endCursorX = x;
        _endCursorY = y;
    }
    
    /// <summary>
    /// 檢查並調整游標位置和偏移量，確保游標在可見區域內
    /// </summary>
    private void AdjustCursorAndOffset()
    {
        // 調用 VimEditor 中的 AdjustCursorAndOffset 方法
        Instance.AdjustCursorPositionAndOffset(Instance.Context.CursorX, Instance.Context.CursorY);
        
        // 更新選取的結束位置
        _endCursorX = Instance.Context.CursorX;
        _endCursorY = Instance.Context.CursorY;
    }
    
    /// <summary>
    /// 切換到普通模式
    /// </summary>
    private void SwitchToNormalMode(List<ConsoleKeyInfo> keys)
    {
        Instance.Mode = new VimInsertMode(Instance);
    }
    
    /// <summary>
    /// 切換到視覺模式
    /// </summary>
    private void SwitchToVisualMode(List<ConsoleKeyInfo> keys)
    {
        Instance.Mode = new VimNormalMode(Instance);
    }
    
    /// <summary>
    /// 向左移動游標
    /// </summary>
    private void MoveCursorLeft(List<ConsoleKeyInfo> keys)
    {
        // 如果啟用了相對行號，則游標的 X 位置不能小於行號區域的寬度
        if (Instance.Context.IsLineNumberVisible)
        {
            // 計算相對行號區域的寬度
            int lineNumberWidth = Instance.Context.GetLineNumberWidth();
            
            // 如果游標已經在最左邊（相對行號區域的右側），則不再向左移動
            if (Instance.Context.CursorX <= lineNumberWidth)
            {
                return;
            }
        }
        
        if (Instance.Context.CursorX > 0)
        {
            Instance.Context.CursorX--;
            AdjustCursorAndOffset();
        }
    }
    
    /// <summary>
    /// 向右移動游標
    /// </summary>
    private void MoveCursorRight(List<ConsoleKeyInfo> keys)
    {
        // 檢查當前行是否存在
        if (Instance.Context.CursorY < Instance.Context.Texts.Count)
        {
            var currentLine = Instance.Context.Texts[Instance.Context.CursorY];
            
            // 獲取當前文本
            string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
            
            // 計算實際索引位置
            int actualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);
            
            // 檢查是否已經到達文本尾部
            if (actualIndex < currentText.Length)
            {
                // 獲取當前字符的寬度
                char currentChar = currentText[actualIndex];
                
                // 檢查是否是最後一個字符
                if (actualIndex == currentText.Length - 1)
                {
                    // 如果是最後一個字符，游標應該停在這個字符上，而不是超出
                    // 不需要移動游標
                }
                else
                {
                    // 如果不是最後一個字符，正常移動游標
                    Instance.Context.CursorX += currentChar.GetCharWidth();
                }
                
                AdjustCursorAndOffset();
            }
        }
    }
    
    /// <summary>
    /// 向上移動游標
    /// </summary>
    private void MoveCursorUp(List<ConsoleKeyInfo> keys)
    {
        if (Instance.Context.CursorY > 0)
        {
            // 保存當前行信息
            var currentLine = Instance.Context.Texts[Instance.Context.CursorY];
            string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
            
            // 檢查游標是否在當前行的最後一個字符上
            // 在標記模式下，判斷游標是否在文本結束位置是通過檢查它是否在最後一個字符上
            int currentActualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);
            bool isAtEndOfCurrentLine = (currentActualIndex == currentText.Length - 1);
            
            // 移動到上一行
            Instance.Context.CursorY--;
            
            // 獲取上一行信息
            var upLine = Instance.Context.Texts[Instance.Context.CursorY];
            string upLineText = new string(upLine.Chars.Select(c => c.Char).ToArray());
            
            // 如果游標在當前行的最後一個字符上，則移動到上一行的最後一個字符上
            if (isAtEndOfCurrentLine && upLineText.Length > 0)
            {
                // 計算上一行最後一個字符的顯示位置
                int displayPosition = 0;
                for (int i = 0; i < upLineText.Length - 1; i++)
                {
                    displayPosition += upLineText[i].GetCharWidth();
                }
                Instance.Context.CursorX = displayPosition;
            }
            // 否則，如果游標X位置超過上一行的長度，則調整到上一行的末尾
            else if (Instance.Context.CursorX > upLineText.GetStringDisplayWidth())
            {
                Instance.Context.CursorX = upLineText.GetStringDisplayWidth();
                // 確保游標不會超出實際文本
                if (upLineText.Length > 0)
                {
                    int adjustedIndex = upLineText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);
                    if (adjustedIndex >= upLineText.Length)
                    {
                        adjustedIndex = upLineText.Length - 1;
                        Instance.Context.CursorX = 0;
                        for (int i = 0; i <= adjustedIndex; i++)
                        {
                            Instance.Context.CursorX += upLineText[i].GetCharWidth();
                        }
                    }
                }
            }
            // 否則保持游標X位置不變
            
            AdjustCursorAndOffset();
        }
    }
    
    /// <summary>
    /// 向下移動游標
    /// </summary>
    private void MoveCursorDown(List<ConsoleKeyInfo> keys)
    {
        if (Instance.Context.CursorY < Instance.Context.Texts.Count - 1)
        {
            // 保存當前行信息
            var currentLine = Instance.Context.Texts[Instance.Context.CursorY];
            string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
            
            // 檢查游標是否在當前行的最後一個字符上
            // 在標記模式下，判斷游標是否在文本結束位置是通過檢查它是否在最後一個字符上
            int currentActualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);
            bool isAtEndOfCurrentLine = (currentActualIndex == currentText.Length - 1);
            
            // 移動到下一行
            Instance.Context.CursorY++;
            
            // 獲取下一行信息
            var downLine = Instance.Context.Texts[Instance.Context.CursorY];
            string downLineText = new string(downLine.Chars.Select(c => c.Char).ToArray());
            
            // 如果游標在當前行的最後一個字符上，則移動到下一行的最後一個字符上
            if (isAtEndOfCurrentLine && downLineText.Length > 0)
            {
                // 計算下一行最後一個字符的顯示位置
                int displayPosition = 0;
                for (int i = 0; i < downLineText.Length - 1; i++)
                {
                    displayPosition += downLineText[i].GetCharWidth();
                }
                Instance.Context.CursorX = displayPosition;
            }
            // 否則，如果游標X位置超過下一行的長度，則調整到下一行的末尾
            else if (Instance.Context.CursorX > downLineText.GetStringDisplayWidth())
            {
                Instance.Context.CursorX = downLineText.GetStringDisplayWidth();
                // 確保游標不會超出實際文本
                if (downLineText.Length > 0)
                {
                    int adjustedIndex = downLineText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);
                    if (adjustedIndex >= downLineText.Length)
                    {
                        adjustedIndex = downLineText.Length - 1;
                        Instance.Context.CursorX = 0;
                        for (int i = 0; i <= adjustedIndex; i++)
                        {
                            Instance.Context.CursorX += downLineText[i].GetCharWidth();
                        }
                    }
                }
            }
            // 否則保持游標X位置不變
            
            AdjustCursorAndOffset();
        }
    }
    
    /// <summary>
    /// 複製選取的文本
    /// </summary>
    private void CopySelectedText(List<ConsoleKeyInfo> keys)
    {
        // 確保起始位置和結束位置有序
        int startY = Math.Min(_startCursorY, _endCursorY);
        int endY = Math.Max(_startCursorY, _endCursorY);
        int startX = _startCursorY < _endCursorY ? _startCursorX : (_startCursorY > _endCursorY ? _endCursorX : Math.Min(_startCursorX, _endCursorX));
        int endX = _startCursorY < _endCursorY ? _endCursorX : (_startCursorY > _endCursorY ? _startCursorX : Math.Max(_startCursorX, _endCursorX));
        
        // 創建剪貼簿緩衝區，如果不存在
        if (Instance.ClipboardBuffers == null)
        {
            Instance.ClipboardBuffers = new List<ConsoleText>();
        }
        
        // 清空剪貼簿
        Instance.ClipboardBuffers.Clear();
        
        // 複製選取的文本到剪貼簿
        if (startY == endY)
        {
            // 如果選取範圍在同一行
            var line = Instance.Context.Texts[startY];
            var text = new ConsoleText();
            
            // 獲取當前行的文本
            string lineText = new string(line.Chars.Select(c => c.Char).ToArray());
            
            // 計算實際索引位置
            int startIndex = lineText.GetStringIndexFromDisplayPosition(startX);
            int endIndex = lineText.GetStringIndexFromDisplayPosition(endX);
            
            // 確保索引有效
            startIndex = Math.Max(0, startIndex);
            endIndex = Math.Min(lineText.Length - 1, endIndex);
            
            // 直接從原始文本中提取選中的部分
            string selectedText = lineText.Substring(startIndex, endIndex - startIndex + 1);
            
            // 設置到剪貼簿
            text.SetText(0, selectedText);
            Instance.ClipboardBuffers.Add(text);
        }
        else
        {
            // 如果選取範圍跨越多行
            for (int y = startY; y <= endY; y++)
            {
                var line = Instance.Context.Texts[y];
                var text = new ConsoleText();
                
                // 獲取當前行的文本
                string lineText = new string(line.Chars.Select(c => c.Char).ToArray());
                
                if (y == startY)
                {
                    // 第一行：從起始位置到行尾
                    int startIndex = lineText.GetStringIndexFromDisplayPosition(startX);
                    startIndex = Math.Max(0, startIndex);
                    
                    // 提取選中的部分
                    string selectedText = lineText.Substring(startIndex);
                    text.SetText(0, selectedText);
                }
                else if (y == endY)
                {
                    // 最後一行：從行首到結束位置
                    int endIndex = lineText.GetStringIndexFromDisplayPosition(endX);
                    endIndex = Math.Min(lineText.Length - 1, endIndex);
                    
                    // 提取選中的部分
                    string selectedText = lineText.Substring(0, endIndex + 1);
                    text.SetText(0, selectedText);
                }
                else
                {
                    // 中間行：整行複製
                    text.SetText(0, lineText);
                }
                
                Instance.ClipboardBuffers.Add(text);
            }
        }
        
        // 複製完成後切換回視覺模式
        Instance.Context.StatusBar.SetText(0, "已複製選取的文本");
        Instance.Context.IsStatusBarVisible = true;
        SwitchToVisualMode(keys);
    }
    
    /// <summary>
    /// 設置游標位置
    /// </summary>
    private void SetCursorPosition()
    {
        // 設置為方塊游標 (DECSCUSR 2)
        Instance.Console.Write("\x1b[2 q");
    }
    
    public void WaitForInput()
    {
        // 設置為方塊游標 (DECSCUSR 2)
        Instance.Console.Write("\x1b[2 q");
        // 高亮顯示選取的文本
        HighlightSelectedText();
        _keyHandler.WaitForInput();
        SetCursorPosition();
    }
    
    /// <summary>
    /// 高亮顯示選取的文本
    /// </summary>
    private void HighlightSelectedText()
    {
        // 確保起始位置和結束位置有序
        int startY = Math.Min(_startCursorY, _endCursorY);
        int endY = Math.Max(_startCursorY, _endCursorY);
        
        // 備份原始文本顏色
        var backupTexts = new Dictionary<int, ColoredChar[]>();
        
        // 對選取範圍內的每一行進行處理
        for (int y = startY; y <= endY; y++)
        {
            if (y >= 0 && y < Instance.Context.Texts.Count)
            {
                var line = Instance.Context.Texts[y];
                
                // 備份原始文本
                backupTexts[y] = line.Chars.ToArray();
                
                // 計算該行的選取範圍
                int startX = 0;
                int endX = line.Chars.Length - 1;
                
                if (y == _startCursorY && y == _endCursorY)
                {
                    // 如果起始和結束在同一行
                    startX = Math.Min(_startCursorX, _endCursorX);
                    endX = Math.Max(_startCursorX, _endCursorX);
                }
                else if (y == _startCursorY)
                {
                    // 如果是起始行
                    startX = _startCursorX;
                }
                else if (y == _endCursorY)
                {
                    // 如果是結束行
                    endX = _endCursorX;
                }
                
                // 高亮顯示選取範圍
                string lineText = new string(line.Chars.Select(c => c.Char).ToArray());
                int startIndex = Math.Min(lineText.GetStringIndexFromDisplayPosition(startX), lineText.Length - 1);
                int endIndex = Math.Min(lineText.GetStringIndexFromDisplayPosition(endX), lineText.Length - 1);
                
                startIndex = Math.Max(0, startIndex);
                endIndex = Math.Max(0, endIndex);
                
                for (int i = startIndex; i <= endIndex && i < line.Chars.Length; i++)
                {
                    if (line.Chars[i].Char != '\0')
                    {
                        // 反轉前景色和背景色
                        line.Chars[i] = new ColoredChar(
                            line.Chars[i].Char,
                            line.Chars[i].BackgroundColor,
                            line.Chars[i].ForegroundColor
                        );
                    }
                }
            }
        }
        
        // 渲染畫面
        Instance.Render();
        
        // 恢復原始文本顏色
        foreach (var entry in backupTexts)
        {
            Instance.Context.Texts[entry.Key].Chars = entry.Value;
        }
    }
} 