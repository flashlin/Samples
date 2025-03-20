namespace VimSharpLib;

/// <summary>
/// 控制台設備介面，抽象化控制台操作
/// </summary>
public interface IConsoleDevice
{
    /// <summary>
    /// 獲取控制台視窗寬度
    /// </summary>
    int WindowWidth { get; }
    
    /// <summary>
    /// 獲取控制台視窗高度
    /// </summary>
    int WindowHeight { get; }
    
    /// <summary>
    /// 設置光標位置
    /// </summary>
    /// <param name="left">左邊距</param>
    /// <param name="top">上邊距</param>
    void SetCursorPosition(int left, int top);
    
    /// <summary>
    /// 寫入文本到控制台
    /// </summary>
    /// <param name="value">要寫入的文本</param>
    void Write(string value);
    
    /// <summary>
    /// 讀取按鍵
    /// </summary>
    /// <param name="intercept">是否攔截按鍵</param>
    /// <returns>按鍵信息</returns>
    ConsoleKeyInfo ReadKey(bool intercept);

    void SetBlockCursor();
} 