namespace VimSharpLib;

/// <summary>
/// 表示單個按鍵模式
/// </summary>
public class UserKeyPattern : IKeyPattern
{
    /// <summary>
    /// 按鍵字符
    /// </summary>
    public char KeyChar { get; set; }
    
    /// <summary>
    /// 是否按下 Ctrl 鍵
    /// </summary>
    public bool IsCtrl { get; set; }
    
    /// <summary>
    /// 是否按下 Alt 鍵
    /// </summary>
    public bool IsAlt { get; set; }
    
    /// <summary>
    /// 檢查提供的按鍵緩衝區是否匹配此模式
    /// </summary>
    /// <param name="keyBuffer">要檢查的按鍵信息緩衝區</param>
    /// <returns>如果匹配則返回 true，否則返回 false</returns>
    public bool IsMatch(List<ConsoleKeyInfo> keyBuffer)
    {
        if (keyBuffer.Count != 1) return false;
        
        var keyInfo = keyBuffer[0];
        return keyInfo.KeyChar == KeyChar && 
               keyInfo.Modifiers.HasFlag(ConsoleModifiers.Control) == IsCtrl && 
               keyInfo.Modifiers.HasFlag(ConsoleModifiers.Alt) == IsAlt;
    }
} 