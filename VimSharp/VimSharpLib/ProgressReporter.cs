namespace VimSharpLib;

/// <summary>
/// 進度報告器，實現 IProgress 介面
/// </summary>
public class ProgressReporter : IProgress
{
    private readonly VimEditor _editor;
    
    /// <summary>
    /// 建構函數
    /// </summary>
    /// <param name="editor">編輯器實例</param>
    public ProgressReporter(VimEditor editor)
    {
        _editor = editor;
    }
    
    /// <summary>
    /// 顯示進度訊息在編輯器的狀態列
    /// </summary>
    /// <param name="message">進度訊息</param>
    public void ShowProgress(string message)
    {
        // 暫存當前狀態
        var originalStatusBarContent = _editor.Context.StatusBar?.ToString() ?? string.Empty;
        
        // 更新狀態列顯示進度訊息
        var statusText = new ConsoleText();
        statusText.SetText(0, message);
        _editor.Context.StatusBar = statusText;
        
        // 設置反色顯示
        for (int i = 0; i < _editor.Context.StatusBar.Chars.Length; i++)
        {
            var c = _editor.Context.StatusBar.Chars[i];
            if (c == ColoredChar.None)
            {
                continue;
            }
            
            char displayChar = (c.Char == '\0') ? ' ' : c.Char;
            _editor.Context.StatusBar.Chars[i] = new ColoredChar(displayChar, ConsoleColor.Black, ConsoleColor.White);
        }
        
        // 重新渲染編輯器以顯示進度訊息
        var screenBuffer = _editor.ScreenBuffer;
        if (screenBuffer != null)
        {
            _editor.Render(screenBuffer);
        }
    }
} 