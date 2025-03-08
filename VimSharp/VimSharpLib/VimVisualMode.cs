namespace VimSharpLib;
using System.Text;

public class VimVisualMode : IVimMode
{
    public required VimEditor Instance { get; set; }
    
    public void WaitForInput()
    {
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