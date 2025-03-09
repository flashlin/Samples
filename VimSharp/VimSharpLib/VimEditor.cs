namespace VimSharpLib;
using System.Text;

public class VimEditor
{
    private readonly IConsoleDevice _console;
    
    public bool IsRunning { get; set; } = true;
    public ConsoleContext Context { get; set; } = new();
    public IVimMode Mode { get; set; } = null!;

    // 添加狀態欄相關屬性
    public bool IsStatusBarVisible { get; set; } = false;
    public string StatusBarText { get; set; } = "";

    public VimEditor() : this(new ConsoleDevice())
    {
    }
    
    public VimEditor(IConsoleDevice console)
    {
        _console = console;
        // 初始化 Mode
        Mode = new VimVisualMode { Instance = this };
        Initialize();
    }

    public void Initialize()
    {
        Context.SetText(0, 0, "Hello, World!");
        
        // 設置 ViewPort 的初始值
        // 默認使用整個控制台視窗，但可以由使用者自定義
        if (Context.ViewPort.Width == 0 || Context.ViewPort.Height == 0)
        {
            Context.ViewPort = new ConsoleRectangle(0, 0, _console.WindowWidth, _console.WindowHeight);
        }
    }

    public void Run()
    {
        while (IsRunning)
        {
            Render();
            WaitForInput();
        }
    }

    public void Render()
    {
        // 計算可見區域的行數
        int visibleLines = Math.Min(Context.ViewPort.Height, Context.Texts.Count - Context.OffsetY);
        
        // 如果狀態欄可見，則減少一行用於顯示狀態欄
        int maxLines = IsStatusBarVisible ? Context.ViewPort.Height - 1 : Context.ViewPort.Height;
        
        // 只繪製可見區域內的行
        for (var i = 0; i < maxLines; i++)
        {
            // 計算實際要繪製的文本行索引
            int textIndex = Context.OffsetY + i;
            
            // 確保索引有效
            if (textIndex >= 0 && textIndex < Context.Texts.Count)
            {
                var text = Context.Texts[textIndex];
                
                // 直接繪製文本，考慮 ViewPort 和偏移量
                RenderText(Context.ViewPort.X, Context.ViewPort.Y + i, text, Context.OffsetX, Context.ViewPort);
            }
            else
            {
                // 如果索引無效（超出文本範圍），繪製空白行
                RenderEmptyLine(Context.ViewPort.X, Context.ViewPort.Y + i, Context.ViewPort.Width);
            }
        }
        
        // 如果狀態欄可見，則繪製狀態欄
        if (IsStatusBarVisible)
        {
            RenderStatusBar();
        }

        // 設置光標位置，考慮偏移量
        int cursorScreenX = Context.CursorX - Context.OffsetX + Context.ViewPort.X;
        int cursorScreenY = Context.CursorY - Context.OffsetY + Context.ViewPort.Y;
        
        // 確保光標在可見區域內
        if (cursorScreenX >= Context.ViewPort.X && 
            cursorScreenX < Context.ViewPort.X + Context.ViewPort.Width &&
            cursorScreenY >= Context.ViewPort.Y && 
            cursorScreenY < Context.ViewPort.Y + Context.ViewPort.Height)
        {
            _console.SetCursorPosition(cursorScreenX, cursorScreenY);
        }
    }
    
    /// <summary>
    /// 繪製文本，考慮 ViewPort 和偏移量
    /// </summary>
    private void RenderText(int x, int y, ConsoleText text, int offset, ConsoleRectangle viewPort)
    {
        // 檢查 Y 座標是否在 ViewPort 範圍內
        if (y < viewPort.Y || y >= viewPort.Y + viewPort.Height)
        {
            return; // Y 座標超出範圍，不繪製
        }

        // 設置光標位置到可見區域的起始位置
        _console.SetCursorPosition(x, y);
        
        // 計算可見區域的寬度
        int visibleWidth = viewPort.Width;
        
        // 創建 StringBuilder 來構建輸出字符串
        var sb = new StringBuilder();
        
        // 計算可見的起始和結束位置
        int startX = Math.Max(0, offset);
        int endX = Math.Min(text.Chars.Length, offset + visibleWidth);
        
        // 計算實際要繪製的字符數量
        int charsToDraw = endX - startX;
        
        // 如果有文本內容在可見範圍內
        if (startX < text.Chars.Length && charsToDraw > 0)
        {
            // 繪製文本內容
            for (int i = startX; i < endX; i++)
            {
                var c = text.Chars[i];
                if (c.Char == '\0')
                {
                    // 如果是空字符，添加一個空格（黑底白字）
                    sb.Append(ColoredChar.Empty.ToAnsiString());
                }
                else
                {
                    sb.Append(c.ToAnsiString());
                }
            }
        }
        
        // 計算需要填充的空白字符數量
        int paddingCount = visibleWidth - charsToDraw;
        
        // 如果需要填充空白字符
        if (paddingCount > 0)
        {
            // 填充空白字符
            for (int i = 0; i < paddingCount; i++)
            {
                sb.Append(ColoredChar.Empty.ToAnsiString());
            }
        }
        
        // 輸出構建好的字符串
        _console.Write(sb.ToString());
    }
    
    /// <summary>
    /// 繪製空白行
    /// </summary>
    private void RenderEmptyLine(int x, int y, int width)
    {
        // 檢查 Y 座標是否在控制台範圍內
        if (y < 0 || y >= _console.WindowHeight)
        {
            return; // Y 座標超出範圍，不繪製
        }

        // 設置光標位置
        _console.SetCursorPosition(x, y);
        
        // 創建 StringBuilder 來構建輸出字符串
        var sb = new StringBuilder();
        
        // 填充空白字符
        for (int i = 0; i < width; i++)
        {
            sb.Append(ColoredChar.Empty.ToAnsiString());
        }
        
        // 輸出構建好的字符串
        _console.Write(sb.ToString());
    }
    
    /// <summary>
    /// 繪製狀態欄
    /// </summary>
    private void RenderStatusBar()
    {
        // 計算狀態欄的位置
        int statusBarY = Context.ViewPort.Y + Context.ViewPort.Height - 1;
        
        // 設置光標位置到狀態欄
        _console.SetCursorPosition(Context.ViewPort.X, statusBarY);
        
        // 創建狀態欄文本
        string statusText = StatusBarText;
        if (string.IsNullOrEmpty(statusText))
        {
            // 如果沒有設置狀態欄文本，則顯示默認信息
            string modeName = Mode.GetType().Name.Replace("Vim", "").Replace("Mode", "");
            statusText = $" {modeName} | Line: {Context.CursorY + 1} | Col: {Context.CursorX + 1} ";
        }
        
        // 確保狀態欄文本不超過視窗寬度
        if (statusText.Length > Context.ViewPort.Width)
        {
            statusText = statusText.Substring(0, Context.ViewPort.Width);
        }
        else if (statusText.Length < Context.ViewPort.Width)
        {
            // 如果狀態欄文本不夠長，則用空格填充
            statusText = statusText.PadRight(Context.ViewPort.Width);
        }
        
        // 創建狀態欄的 ConsoleText
        var statusBarText = new ConsoleText();
        statusBarText.SetWidth(statusText.Length);
        
        // 填充狀態欄文本
        for (int i = 0; i < statusText.Length; i++)
        {
            // 使用反色顯示狀態欄
            statusBarText.Chars[i] = new ColoredChar(statusText[i], ConsoleColor.Black, ConsoleColor.White);
        }
        
        // 繪製狀態欄
        RenderText(Context.ViewPort.X, statusBarY, statusBarText, 0, Context.ViewPort);
    }

    public void WaitForInput()
    {
        Mode.WaitForInput();
    }
    
    /// <summary>
    /// 手動調整水平偏移量
    /// </summary>
    /// <param name="offsetX">要設置的水平偏移量</param>
    public void SetHorizontalOffset(int offsetX)
    {
        Context.OffsetX = Math.Max(0, offsetX);
    }
    
    /// <summary>
    /// 手動調整垂直偏移量
    /// </summary>
    /// <param name="offsetY">要設置的垂直偏移量</param>
    public void SetVerticalOffset(int offsetY)
    {
        Context.OffsetY = Math.Max(0, offsetY);
    }
    
    /// <summary>
    /// 手動滾動視圖
    /// </summary>
    /// <param name="deltaX">水平滾動量</param>
    /// <param name="deltaY">垂直滾動量</param>
    public void Scroll(int deltaX, int deltaY)
    {
        Context.OffsetX = Math.Max(0, Context.OffsetX + deltaX);
        Context.OffsetY = Math.Max(0, Context.OffsetY + deltaY);
    }
    
    /// <summary>
    /// 獲取控制台設備
    /// </summary>
    /// <returns>控制台設備</returns>
    public IConsoleDevice GetConsoleDevice()
    {
        return _console;
    }
}