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
    
    public void WaitForInput()
    {
        // 設置游標形狀
        Console.Write("\x1b[2 q");
        
        var keyInfo = Console.ReadKey(intercept: true);
        
        if(keyInfo.Key == ConsoleKey.I)
        {
            Instance.Mode = new VimNormalMode { Instance = Instance };
            return;
        }
        
        if(keyInfo.Key == ConsoleKey.Q)
        {
            Instance.IsRunning = false;
            return;
        }
        
        // 處理方向鍵和特殊鍵
        if (keyInfo.Key == ConsoleKey.LeftArrow && Instance.Context.CursorX > 0)
        {
            Instance.Context.CursorX--;
            AdjustCursorAndOffset();
        }
        else if (keyInfo.Key == ConsoleKey.RightArrow)
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
                if (actualIndex < currentText.Length - 1)
                {
                    // 獲取當前字符的寬度
                    char currentChar = currentText[actualIndex];
                    Instance.Context.CursorX += currentChar.GetCharWidth();
                    AdjustCursorAndOffset();
                }
            }
        }
        else if (keyInfo.Key == ConsoleKey.UpArrow && Instance.Context.CursorY > 0)
        {
            Instance.Context.CursorY--;
            AdjustCursorAndOffset();
        }
        else if (keyInfo.Key == ConsoleKey.DownArrow && Instance.Context.CursorY < Instance.Context.Texts.Count - 1)
        {
            Instance.Context.CursorY++;
            AdjustCursorAndOffset();
        }
        
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
} 