namespace VimSharpLib;
using System.Text;
using System.Linq;

public class VimNormalMode : IVimMode
{
    public required VimEditor Instance { get; set; }
    
    /// <summary>
    /// 檢查並調整游標位置和偏移量，確保游標在可見區域內
    /// </summary>
    private void AdjustCursorAndOffset()
    {
        // 計算行號區域寬度
        int lineNumberWidth = Instance.IsRelativeLineNumber ? Instance.CalculateLineNumberWidth() : 0;
        
        // 考慮行號區域寬度的有效游標水平位置
        int effectiveCursorX = Instance.Context.CursorX;
        
        // 確保游標 X 不低於行號區域寬度
        if (Instance.IsRelativeLineNumber && effectiveCursorX < lineNumberWidth)
        {
            Instance.Context.CursorX = lineNumberWidth;
            effectiveCursorX = lineNumberWidth;
        }
        
        // 計算游標在屏幕上的位置
        int cursorScreenX = effectiveCursorX - Instance.Context.OffsetX;
        int cursorScreenY = Instance.Context.CursorY - Instance.Context.OffsetY;
        
        // 計算可見區域的有效高度（考慮狀態欄）
        int effectiveViewPortHeight = Instance.Context.IsStatusBarVisible
            ? Instance.Context.ViewPort.Height - 1
            : Instance.Context.ViewPort.Height;
        
        // 檢查游標是否超出右邊界
        if (cursorScreenX >= Instance.Context.ViewPort.Width)
        {
            // 調整水平偏移量，使游標位於可見區域的右邊界
            Instance.Context.OffsetX = effectiveCursorX - Instance.Context.ViewPort.Width + 1;
        }
        // 檢查游標是否超出左邊界
        else if (cursorScreenX < 0)
        {
            // 調整水平偏移量，使游標位於可見區域的左邊界
            Instance.Context.OffsetX = effectiveCursorX;
        }
        
        // 檢查游標是否超出下邊界
        if (cursorScreenY >= effectiveViewPortHeight)
        {
            // 調整垂直偏移量，使游標位於可見區域的下邊界
            Instance.Context.OffsetY = Instance.Context.CursorY - effectiveViewPortHeight + 1;
        }
        // 檢查游標是否超出上邊界
        else if (cursorScreenY < 0)
        {
            // 調整垂直偏移量，使游標位於可見區域的上邊界
            Instance.Context.OffsetY = Instance.Context.CursorY;
        }
        
        // 確保游標在文本範圍內
        Instance.Context.CursorY = Math.Min(Instance.Context.CursorY, Instance.Context.Texts.Count - 1);
        Instance.Context.CursorY = Math.Max(0, Instance.Context.CursorY);
        
        // 確保當前行存在
        if (Instance.Context.Texts.Count == 0)
        {
            Instance.Context.Texts.Add(new ConsoleText());
        }
        
        // 確保游標水平位置在當前行文本範圍內
        var currentLine = Instance.Context.Texts[Instance.Context.CursorY];
        string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
        int textWidth = currentText.GetStringDisplayWidth();
        
        // 在普通模式下，游標可以停在最後一個字符上
        Instance.Context.CursorX = Math.Min(Instance.Context.CursorX, Math.Max(lineNumberWidth, textWidth));
    }
    
    /// <summary>
    /// 切換到視覺模式
    /// </summary>
    private void SwitchToVisualMode()
    {
        // 如果游標不在行首，則向左移動一格
        if (Instance.Context.CursorX > 0)
        {
            // 獲取當前行
            var currentLine = Instance.Context.Texts[Instance.Context.CursorY];
            string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
            
            // 計算實際索引位置
            int actualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);
            
            // 如果不是在行首，則向左移動一格
            if (actualIndex > 0)
            {
                // 獲取前一個字符的寬度
                char prevChar = currentText[actualIndex - 1];
                Instance.Context.CursorX -= prevChar.GetCharWidth();
            }
        }
        
        // 切換到視覺模式
        Instance.Mode = new VimVisualMode { Instance = Instance };
    }
    
    /// <summary>
    /// 處理退格鍵
    /// </summary>
    private void HandleBackspace()
    {
        if (Instance.Context.CursorX > 0)
        {
            // 獲取當前行
            var currentLine = Instance.Context.Texts[Instance.Context.CursorY];

            // 獲取當前文本
            string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());

            // 計算實際索引位置
            int actualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);

            if (actualIndex > 0)
            {
                // 獲取要刪除的字符
                char charToDelete = currentText[actualIndex - 1];

                // 刪除字符
                string newText = currentText.Remove(actualIndex - 1, 1);

                // 更新文本
                currentLine.SetText(0, newText);

                // 移動光標（考慮中文字符寬度）
                Instance.Context.CursorX -= charToDelete.GetCharWidth();

                // 清除屏幕並重新渲染整行（對於 Backspace，我們需要重新渲染整行）
                Instance.Render();
            }
        }
    }
    
    /// <summary>
    /// 向左移動游標
    /// </summary>
    private void MoveCursorLeft()
    {
        // 如果啟用了相對行號，則游標的 X 位置不能小於行號區域的寬度
        if (Instance.IsRelativeLineNumber)
        {
            // 計算相對行號區域的寬度
            int lineNumberWidth = Instance.CalculateLineNumberWidth();
            
            // 如果游標已經在最左邊（相對行號區域的右側），則不再向左移動
            if (Instance.Context.CursorX <= lineNumberWidth)
            {
                return;
            }
        }
        
        if (Instance.Context.CursorX > 0)
        {
            // 獲取當前文本
            var currentLine = Instance.Context.Texts[Instance.Context.CursorY];
            string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());

            // 計算實際索引位置
            int actualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);

            if (actualIndex > 0)
            {
                // 獲取前一個字符的寬度
                char prevChar = currentText[actualIndex - 1];
                int newCursorX = Instance.Context.CursorX - prevChar.GetCharWidth();
                
                // 如果啟用了相對行號，確保游標的 X 位置不會小於行號區域的寬度
                if (Instance.IsRelativeLineNumber)
                {
                    int lineNumberWidth = Instance.CalculateLineNumberWidth();
                    if (newCursorX < lineNumberWidth)
                    {
                        Instance.Context.CursorX = lineNumberWidth;
                    }
                    else
                    {
                        Instance.Context.CursorX = newCursorX;
                    }
                }
                else
                {
                    Instance.Context.CursorX = newCursorX;
                }
            }
        }
        
        // 檢查並調整游標位置和偏移量
        AdjustCursorAndOffset();
    }
    
    /// <summary>
    /// 向右移動游標
    /// </summary>
    private void MoveCursorRight()
    {
        var currentLineForRight = Instance.Context.Texts[Instance.Context.CursorY];
        string textForRight = new string(currentLineForRight.Chars.Select(c => c.Char).ToArray());

        // 計算實際索引位置
        int actualIndexForRight = textForRight.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);

        // 檢查並跳過 '\0' 字符
        while (actualIndexForRight < textForRight.Length && textForRight[actualIndexForRight] == '\0')
        {
            actualIndexForRight++;
        }

        if (actualIndexForRight < textForRight.Length)
        {
            // 獲取當前字符的寬度
            char currentChar = textForRight[actualIndexForRight];
            
            // 移動游標
            Instance.Context.CursorX += currentChar.GetCharWidth();
        }
        else if (actualIndexForRight == textForRight.Length)
        {
            // 允許游標移動到最後一個字符後面
            Instance.Context.CursorX = textForRight.GetStringDisplayWidth() + 1;
        }
        
        // 檢查並調整游標位置和偏移量
        AdjustCursorAndOffset();
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
            
            // 檢查游標是否在當前行的文本結束位置
            // 在普通模式下，判斷游標是否在文本結束位置是通過檢查它是否在文本的末尾
            bool isAtEndOfCurrentLine = (Instance.Context.CursorX >= currentText.GetStringDisplayWidth());
            
            // 移動到上一行
            Instance.Context.CursorY--;
            
            // 獲取上一行信息
            var upLine = Instance.Context.Texts[Instance.Context.CursorY];
            string upLineText = new string(upLine.Chars.Select(c => c.Char).ToArray());
            
            // 如果游標在當前行的文本結束位置，則移動到上一行的文本結束位置
            if (isAtEndOfCurrentLine)
            {
                Instance.Context.CursorX = upLineText.GetStringDisplayWidth();
            }
            // 否則，如果游標X位置超過上一行的長度，則調整到上一行的末尾
            else if (Instance.Context.CursorX > upLineText.GetStringDisplayWidth())
            {
                Instance.Context.CursorX = upLineText.GetStringDisplayWidth();
            }
            // 否則保持游標X位置不變
            
            // 檢查並調整游標位置和偏移量
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
            
            // 檢查游標是否在當前行的文本結束位置
            // 在普通模式下，判斷游標是否在文本結束位置是通過檢查它是否在文本的末尾
            bool isAtEndOfCurrentLine = (Instance.Context.CursorX >= currentText.GetStringDisplayWidth());
            
            // 移動到下一行
            Instance.Context.CursorY++;
            
            // 獲取下一行信息
            var downLine = Instance.Context.Texts[Instance.Context.CursorY];
            string downLineText = new string(downLine.Chars.Select(c => c.Char).ToArray());
            
            // 如果游標在當前行的文本結束位置，則移動到下一行的文本結束位置
            if (isAtEndOfCurrentLine)
            {
                Instance.Context.CursorX = downLineText.GetStringDisplayWidth();
            }
            // 否則，如果游標X位置超過下一行的長度，則調整到下一行的末尾
            else if (Instance.Context.CursorX > downLineText.GetStringDisplayWidth())
            {
                Instance.Context.CursorX = downLineText.GetStringDisplayWidth();
            }
            // 否則保持游標X位置不變
            
            // 檢查並調整游標位置和偏移量
            AdjustCursorAndOffset();
        }
    }
    
    /// <summary>
    /// 處理 Enter 鍵
    /// </summary>
    private void HandleEnterKey()
    {
        // 獲取當前行
        var enterCurrentLine = Instance.Context.Texts[Instance.Context.CursorY];
        string enterCurrentText = new string(enterCurrentLine.Chars.Select(c => c.Char).ToArray());
        
        // 計算實際索引位置
        int enterActualIndex = enterCurrentText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);
        
        // 檢查游標後面是否有內容
        string remainingText = "";
        if (enterActualIndex < enterCurrentText.Length)
        {
            // 獲取游標後面的內容
            remainingText = enterCurrentText.Substring(enterActualIndex);
            
            // 修改當前行，只保留游標前面的內容
            string newCurrentText = enterCurrentText.Substring(0, enterActualIndex);
            enterCurrentLine.SetText(0, newCurrentText);
        }

        // 在當前行後插入新行
        Instance.Context.CursorY++;
        Instance.Context.CursorX = 0;
        
        // 確保新行存在
        if (Instance.Context.Texts.Count <= Instance.Context.CursorY)
        {
            Instance.Context.Texts.Add(new ConsoleText());
        }
        
        // 如果有剩餘內容，設置到新行
        if (!string.IsNullOrEmpty(remainingText))
        {
            Instance.Context.Texts[Instance.Context.CursorY].SetText(0, remainingText);
        }
        
        // 檢查並調整游標位置和偏移量
        AdjustCursorAndOffset();
    }
    
    /// <summary>
    /// 處理一般字符輸入
    /// </summary>
    private void HandleCharInput(char keyChar)
    {
        if (char.IsLetterOrDigit(keyChar) || char.IsPunctuation(keyChar) || char.IsWhiteSpace(keyChar))
        {
            var currentLine = Instance.Context.Texts[Instance.Context.CursorY];
            
            // 獲取當前文本
            string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
            
            // 計算實際索引位置
            int actualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);
            
            // 在實際索引位置插入字符
            string newText = currentText.Insert(actualIndex, keyChar.ToString());
            
            // 更新文本
            currentLine.SetText(0, newText);
            
            // 移動光標（考慮中文字符寬度）
            Instance.Context.CursorX += keyChar.GetCharWidth();
            
            // 如果游標位於文本末尾，確保它停在最後一個字符上
            string updatedText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
            int updatedActualIndex = updatedText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);
            
            if (updatedActualIndex > updatedText.Length)
            {
                // 調整游標位置到最後一個字符
                int lastCharIndex = updatedText.Length - 1;
                if (lastCharIndex >= 0)
                {
                    int displayPosition = 0;
                    for (int i = 0; i <= lastCharIndex; i++)
                    {
                        displayPosition += updatedText[i].GetCharWidth();
                    }
                    Instance.Context.CursorX = displayPosition;
                }
            }
        }
        
        // 檢查並調整游標位置和偏移量
        AdjustCursorAndOffset();
    }

    public void WaitForInput()
    {
        // 設置為垂直線游標 (DECSCUSR 6)
        Instance.GetConsoleDevice().Write("\x1b[6 q");
        
        var keyInfo = Instance.GetConsoleDevice().ReadKey(intercept: true);

        // 確保當前行存在
        if (Instance.Context.Texts.Count <= Instance.Context.CursorY)
        {
            Instance.Context.Texts.Add(new ConsoleText());
        }

        switch (keyInfo.Key)
        {
            case ConsoleKey.Escape:
                SwitchToVisualMode();
                break;

            case ConsoleKey.Backspace:
                HandleBackspace();
                break;
                
            case ConsoleKey.LeftArrow:
                MoveCursorLeft();
                break;
                
            case ConsoleKey.RightArrow:
                MoveCursorRight();
                break;
                
            case ConsoleKey.UpArrow:
                MoveCursorUp();
                break;
                
            case ConsoleKey.DownArrow:
                MoveCursorDown();
                break;
                
            case ConsoleKey.Enter:
                HandleEnterKey();
                break;
                
            default:
                HandleCharInput(keyInfo.KeyChar);
                break;
        }
    }
}