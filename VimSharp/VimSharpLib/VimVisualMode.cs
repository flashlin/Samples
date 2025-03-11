namespace VimSharpLib;
using System.Text;
using System.Linq;
using System;
using System.Collections.Generic;

public class VimVisualMode : IVimMode
{
    public required VimEditor Instance { get; set; }
    
    private Dictionary<IKeyPattern, Action> _keyPatterns = new();
    private List<ConsoleKey> _keyBuffer;
    
    public VimVisualMode()
    {
        _keyBuffer = new List<ConsoleKey>();
        InitializeKeyPatterns();
    }
    
    private void InitializeKeyPatterns()
    {
        _keyPatterns = new Dictionary<IKeyPattern, Action>
        {
            { new ConsoleKeyPattern(ConsoleKey.I), SwitchToNormalMode },
            { new ConsoleKeyPattern(ConsoleKey.A), HandleAKey },
            { new ConsoleKeyPattern(ConsoleKey.Q), QuitEditor },
            { new ConsoleKeyPattern(ConsoleKey.LeftArrow), MoveCursorLeft },
            { new ConsoleKeyPattern(ConsoleKey.RightArrow), MoveCursorRight },
            { new ConsoleKeyPattern(ConsoleKey.UpArrow), MoveCursorUp },
            { new ConsoleKeyPattern(ConsoleKey.DownArrow), MoveCursorDown },
            { new ConsoleKeyPattern(ConsoleKey.Enter), HandleEnterKey },
            { new ConsoleKeyPattern(ConsoleKey.V), SwitchToMarkMode },
            { new ConsoleKeyPattern(ConsoleKey.P), HandlePasteAfterCursor },
        };
    }
    
    /// <summary>
    /// 檢查並調整游標位置和偏移量，確保游標在可見區域內
    /// </summary>
    private void AdjustCursorAndOffset()
    {
        // 計算游標在屏幕上的位置
        int cursorScreenX = Instance.Context.CursorX - Instance.Context.OffsetX;
        int cursorScreenY = Instance.Context.CursorY - Instance.Context.OffsetY;
        
        // 檢查游標是否超出右邊界
        if (cursorScreenX >= Instance.Context.ViewPort.Width)
        {
            // 調整水平偏移量，使游標位於可見區域的右邊界
            Instance.Context.OffsetX = Instance.Context.CursorX - Instance.Context.ViewPort.Width + 1;
        }
        // 檢查游標是否超出左邊界
        else if (cursorScreenX < 0)
        {
            // 調整水平偏移量，使游標位於可見區域的左邊界
            Instance.Context.OffsetX = Instance.Context.CursorX;
        }
        
        // 檢查游標是否超出下邊界
        if (cursorScreenY >= Instance.Context.ViewPort.Height)
        {
            // 調整垂直偏移量，使游標位於可見區域的下邊界
            Instance.Context.OffsetY = Instance.Context.CursorY - Instance.Context.ViewPort.Height + 1;
        }
        // 檢查游標是否超出上邊界
        else if (cursorScreenY < 0)
        {
            // 調整垂直偏移量，使游標位於可見區域的上邊界
            Instance.Context.OffsetY = Instance.Context.CursorY;
        }
    }
    
    /// <summary>
    /// 切換到普通模式
    /// </summary>
    private void SwitchToNormalMode()
    {
        Instance.Mode = new VimNormalMode { Instance = Instance };
    }
    
    /// <summary>
    /// 退出編輯器
    /// </summary>
    private void QuitEditor()
    {
        Instance.IsRunning = false;
    }
    
    /// <summary>
    /// 向左移動游標
    /// </summary>
    private void MoveCursorLeft()
    {
        if (Instance.Context.CursorX > 0)
        {
            Instance.Context.CursorX--;
            AdjustCursorAndOffset();
        }
    }
    
    /// <summary>
    /// 向右移動游標
    /// </summary>
    private void MoveCursorRight()
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
    private void MoveCursorUp()
    {
        if (Instance.Context.CursorY > 0)
        {
            // 保存當前行信息
            var currentLine = Instance.Context.Texts[Instance.Context.CursorY];
            string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
            
            // 檢查游標是否在當前行的最後一個字符上
            // 在視覺模式下，判斷游標是否在文本結束位置是通過檢查它是否在最後一個字符上
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
    private void MoveCursorDown()
    {
        if (Instance.Context.CursorY < Instance.Context.Texts.Count - 1)
        {
            // 保存當前行信息
            var currentLine = Instance.Context.Texts[Instance.Context.CursorY];
            string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
            
            // 檢查游標是否在當前行的最後一個字符上
            // 在視覺模式下，判斷游標是否在文本結束位置是通過檢查它是否在最後一個字符上
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
    /// 處理 Enter 鍵
    /// </summary>
    private void HandleEnterKey()
    {
        // 在視覺模式下，Enter 鍵只移動游標，不修改文本
        // 移動到下一行的開頭
        Instance.Context.CursorY++;
        Instance.Context.CursorX = 0;
        
        // 雖然視覺模式是僅讀取模式，但我們仍然允許添加空行以便瀏覽
        // 這不會修改現有文本內容，只是為了確保游標可以移動到文本末尾之後
        if (Instance.Context.Texts.Count <= Instance.Context.CursorY)
        {
            Instance.Context.Texts.Add(new ConsoleText());
        }
        
        // 檢查並調整游標位置和偏移量
        AdjustCursorAndOffset();
    }
    
    /// <summary>
    /// 設置游標位置
    /// </summary>
    private void SetCursorPosition()
    {
        // 設置光標位置，考慮偏移量但不調整 ViewPort
        int cursorScreenX = Instance.Context.CursorX - Instance.Context.OffsetX + Instance.Context.ViewPort.X;
        int cursorScreenY = Instance.Context.CursorY - Instance.Context.OffsetY + Instance.Context.ViewPort.Y;
        
        // 確保光標在可見區域內
        if (cursorScreenX >= Instance.Context.ViewPort.X && 
            cursorScreenX < Instance.Context.ViewPort.X + Instance.Context.ViewPort.Width &&
            cursorScreenY >= Instance.Context.ViewPort.Y && 
            cursorScreenY < Instance.Context.ViewPort.Y + Instance.Context.ViewPort.Height)
        {
            Instance.GetConsoleDevice().SetCursorPosition(cursorScreenX, cursorScreenY);
        }
    }
    
    /// <summary>
    /// 處理 A 鍵：向右移動游標後切換到普通模式
    /// </summary>
    private void HandleAKey()
    {
        // 獲取當前行信息
        var currentLine = Instance.Context.Texts[Instance.Context.CursorY];
        string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
        
        // 計算實際索引位置
        int actualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);
        
        // 檢查並跳過 '\0' 字符
        while (actualIndex < currentText.Length && currentText[actualIndex] == '\0')
        {
            actualIndex++;
        }

        // 如果不是在文本末尾，則向右移動一個位置
        if (actualIndex < currentText.Length)
        {
            // 獲取當前字符的寬度
            char currentChar = currentText[actualIndex];
            Instance.Context.CursorX += currentChar.GetCharWidth();
        }
        else if (actualIndex == currentText.Length)
        {
            // 允許游標移動到最後一個字符後面
            Instance.Context.CursorX = currentText.GetStringDisplayWidth() + 1;
        }
        
        // 切換到普通模式
        var normalMode = new VimNormalMode { Instance = Instance };
        Instance.Mode = normalMode;
        
        // 調整偏移量
        AdjustCursorAndOffset();
    }
    
    /// <summary>
    /// 切換到標記模式
    /// </summary>
    private void SwitchToMarkMode()
    {
        var markMode = new VimMarkMode { Instance = Instance };
        markMode.SetStartPosition(Instance.Context.CursorX, Instance.Context.CursorY);
        Instance.Mode = markMode;
    }
    
    /// <summary>
    /// 處理小寫 p 鍵：從游標右邊位置插入剪貼簿內容
    /// </summary>
    private void HandlePasteAfterCursor()
    {
        // 檢查剪貼簿是否有內容
        if (Instance.ClipboardBuffers == null || Instance.ClipboardBuffers.Count == 0)
        {
            Instance.StatusBarText = "剪貼簿為空";
            Instance.IsStatusBarVisible = true;
            return;
        }

        // 確保當前行存在
        if (Instance.Context.CursorY >= Instance.Context.Texts.Count)
        {
            Instance.Context.Texts.Add(new ConsoleText());
        }

        // 獲取當前行
        var currentLine = Instance.Context.Texts[Instance.Context.CursorY];
        string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
        
        // 計算實際索引位置
        int actualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);
        
        // 如果剪貼簿只有一行內容
        if (Instance.ClipboardBuffers.Count == 1)
        {
            // 獲取剪貼簿內容
            var clipboardText = Instance.ClipboardBuffers[0];
            string clipboardContent = new string(clipboardText.Chars.Select(c => c.Char).ToArray());
            
            // 在當前位置插入剪貼簿內容
            string newText = currentText.Insert(actualIndex, clipboardContent);
            currentLine.SetText(0, newText);
            
            // 移動游標到插入內容的結尾
            int newCursorX = 0;
            for (int i = 0; i < actualIndex + clipboardContent.Length; i++)
            {
                if (i < newText.Length)
                {
                    newCursorX += newText[i].GetCharWidth();
                }
            }
            Instance.Context.CursorX = newCursorX;
        }
        else
        {
            // 如果剪貼簿有多行內容
            
            // 處理第一行：將剪貼簿第一行插入到當前行游標位置後
            var firstClipboardLine = Instance.ClipboardBuffers[0];
            string firstClipboardContent = new string(firstClipboardLine.Chars.Select(c => c.Char).ToArray());
            
            // 分割當前行
            string beforeCursor = currentText.Substring(0, actualIndex);
            string afterCursor = actualIndex < currentText.Length ? currentText.Substring(actualIndex) : "";
            
            // 更新當前行：前半部分 + 剪貼簿第一行
            string newCurrentLineText = beforeCursor + firstClipboardContent;
            currentLine.SetText(0, newCurrentLineText);
            
            // 插入剪貼簿中間行
            for (int i = 1; i < Instance.ClipboardBuffers.Count - 1; i++)
            {
                var clipboardLine = Instance.ClipboardBuffers[i];
                string clipboardContent = new string(clipboardLine.Chars.Select(c => c.Char).ToArray());
                
                // 在當前行後插入新行
                Instance.Context.CursorY++;
                
                // 確保新行存在
                if (Instance.Context.Texts.Count <= Instance.Context.CursorY)
                {
                    Instance.Context.Texts.Add(new ConsoleText());
                }
                else
                {
                    // 在當前位置插入新行
                    Instance.Context.Texts.Insert(Instance.Context.CursorY, new ConsoleText());
                }
                
                // 設置新行內容
                Instance.Context.Texts[Instance.Context.CursorY].SetText(0, clipboardContent);
            }
            
            // 處理最後一行：剪貼簿最後一行 + 當前行游標後的內容
            if (Instance.ClipboardBuffers.Count > 1)
            {
                var lastClipboardLine = Instance.ClipboardBuffers[Instance.ClipboardBuffers.Count - 1];
                string lastClipboardContent = new string(lastClipboardLine.Chars.Select(c => c.Char).ToArray());
                
                // 在當前行後插入新行
                Instance.Context.CursorY++;
                
                // 確保新行存在
                if (Instance.Context.Texts.Count <= Instance.Context.CursorY)
                {
                    Instance.Context.Texts.Add(new ConsoleText());
                }
                else
                {
                    // 在當前位置插入新行
                    Instance.Context.Texts.Insert(Instance.Context.CursorY, new ConsoleText());
                }
                
                // 設置新行內容：剪貼簿最後一行 + 原游標後的內容
                string newLastLineText = lastClipboardContent + afterCursor;
                Instance.Context.Texts[Instance.Context.CursorY].SetText(0, newLastLineText);
                
                // 設置游標位置到最後一行的剪貼簿內容結尾處
                Instance.Context.CursorX = 0;
                for (int i = 0; i < lastClipboardContent.Length; i++)
                {
                    Instance.Context.CursorX += lastClipboardContent[i].GetCharWidth();
                }
            }
        }
        
        // 調整游標位置和偏移量
        AdjustCursorAndOffset();
        
        // 切換到普通模式
        Instance.Mode = new VimNormalMode { Instance = Instance };
        
        // 顯示狀態欄消息
        Instance.StatusBarText = "已貼上剪貼簿內容";
        Instance.IsStatusBarVisible = true;
    }
    
    public void WaitForInput()
    {
        // 設置為方塊游標 (DECSCUSR 2)
        Instance.GetConsoleDevice().Write("\x1b[2 q");
        
        var keyInfo = Instance.GetConsoleDevice().ReadKey(intercept: true);
        
        // 將按鍵添加到緩衝區
        _keyBuffer.Add(keyInfo.Key);
        
        // 計算匹配的模式數量
        int matchCount = 0;
        IKeyPattern? matchedPattern = null;
        
        foreach (var pattern in _keyPatterns.Keys)
        {
            if (pattern.IsMatch(_keyBuffer))
            {
                matchCount++;
                matchedPattern = pattern;
            }
        }
        
        // 如果只有一個模式匹配，執行對應的操作
        if (matchCount == 1 && matchedPattern != null)
        {
            _keyPatterns[matchedPattern].Invoke();
            _keyBuffer.Clear();
        }
        // 如果沒有模式匹配，但緩衝區已經達到一定長度，清除緩衝區
        // 這裡我們設置一個合理的最大長度，例如 5
        else if (matchCount == 0 && _keyBuffer.Count >= 5)
        {
            _keyBuffer.Clear();
        }
        // 如果有多個模式匹配，不執行任何操作，等待更多按鍵輸入
        
        SetCursorPosition();
    }
} 