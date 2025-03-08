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
        
        // 設置光標位置
        Console.SetCursorPosition(Instance.Context.CursorX, Instance.Context.CursorY);
    }
} 