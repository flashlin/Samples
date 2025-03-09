namespace VimSharpLib;
using System.Text;
using System.Linq;

public class VimVisualMode : IVimMode
{
    public required VimEditor Instance { get; set; }
    
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
            int currentActualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);
            
            // 檢查游標是否在當前行的最後一個字符上
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
                for (int i = 0; i < upLineText.Length; i++)
                {
                    displayPosition += upLineText[i].GetCharWidth();
                }
                Instance.Context.CursorX = displayPosition;
            }
            // 否則，如果游標X位置超過上一行的長度，則調整到上一行的末尾
            else if (Instance.Context.CursorX > upLineText.GetStringDisplayWidth())
            {
                Instance.Context.CursorX = upLineText.GetStringDisplayWidth();
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
            int currentActualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);
            
            // 檢查游標是否在當前行的最後一個字符上
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
                for (int i = 0; i < downLineText.Length; i++)
                {
                    displayPosition += downLineText[i].GetCharWidth();
                }
                Instance.Context.CursorX = displayPosition;
            }
            // 否則，如果游標X位置超過下一行的長度，則調整到下一行的末尾
            else if (Instance.Context.CursorX > downLineText.GetStringDisplayWidth())
            {
                Instance.Context.CursorX = downLineText.GetStringDisplayWidth();
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
            Console.SetCursorPosition(cursorScreenX, cursorScreenY);
        }
    }
    
    public void WaitForInput()
    {
        // 設置為方塊游標 (DECSCUSR 2)
        Console.Write("\x1b[2 q");
        
        var keyInfo = Console.ReadKey(intercept: true);
        
        switch (keyInfo.Key)
        {
            case ConsoleKey.I:
                SwitchToNormalMode();
                break;
                
            case ConsoleKey.Q:
                QuitEditor();
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
        }
        
        SetCursorPosition();
    }
} 