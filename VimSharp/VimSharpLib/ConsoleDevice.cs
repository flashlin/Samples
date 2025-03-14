namespace VimSharpLib;

/// <summary>
/// 控制台設備實現，封裝 System.Console
/// </summary>
public class ConsoleDevice : IConsoleDevice
{
    /// <summary>
    /// 獲取控制台視窗寬度
    /// </summary>
    public int WindowWidth => Console.WindowWidth;
    
    /// <summary>
    /// 獲取控制台視窗高度
    /// </summary>
    public int WindowHeight => Console.WindowHeight;
    
    /// <summary>
    /// 設置光標位置
    /// </summary>
    /// <param name="left">左邊距</param>
    /// <param name="top">上邊距</param>
    public void SetCursorPosition(int left, int top)
    {
        Console.Write($"\x1b[{top+1};{left+1}H");
    }
    
    /// <summary>
    /// 寫入文本到控制台
    /// </summary>
    /// <param name="value">要寫入的文本</param>
    public void Write(string value)
    {
        Console.Write(value);
    }
    
    /// <summary>
    /// 讀取按鍵
    /// </summary>
    /// <param name="intercept">是否攔截按鍵</param>
    /// <returns>按鍵信息</returns>
    public ConsoleKeyInfo ReadKey(bool intercept)
    {
        return Console.ReadKey(intercept);
    }
} 