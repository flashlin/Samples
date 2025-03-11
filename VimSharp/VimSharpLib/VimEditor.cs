namespace VimSharpLib;
using System.Text;
using System.Collections.Generic;

public class VimEditor
{
    private readonly IConsoleDevice _console;
    
    public bool IsRunning { get; set; } = true;
    public ConsoleContext Context { get; set; } = new();
    public IVimMode Mode { get; set; } = null!;

    // 添加狀態欄相關屬性
    public bool IsStatusBarVisible { get; set; } = false;
    public string StatusBarText { get; set; } = "";
    
    // 添加剪貼簿緩衝區
    public List<ConsoleText> ClipboardBuffers { get; set; } = [];

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
        // 隱藏游標
        _console.Write("\x1b[?25l");
        
        // 計算文本總行數
        int totalLines = Context.Texts.Count;
        
        // 計算相對行號的最大值（上下行數的最大值）
        int maxRelativeLineNumber = Math.Max(Context.CursorY, totalLines - Context.CursorY - 1);
        
        // 計算行號區域寬度所需的位數
        int lineNumberDigits = maxRelativeLineNumber > 0 ? (int)Math.Log10(maxRelativeLineNumber) + 1 : 1;
        
        // 行號區域寬度 = 位數 + 1 (用於間隔)
        int lineNumberWidth = lineNumberDigits + 1;
        
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
                
                // 計算相對行號
                int relativeLineNumber = 0;
                bool isCurrentLine = (textIndex == Context.CursorY);
                
                if (!isCurrentLine)
                {
                    // 如果不是當前行，計算相對行號
                    relativeLineNumber = Math.Abs(textIndex - Context.CursorY);
                }
                
                // 繪製行號
                RenderLineNumber(Context.ViewPort.X, Context.ViewPort.Y + i, relativeLineNumber, lineNumberDigits, isCurrentLine);
                
                // 直接繪製文本，考慮 ViewPort、偏移量和行號區域
                RenderText(Context.ViewPort.X + lineNumberWidth, Context.ViewPort.Y + i, text, Context.OffsetX, Context.ViewPort, lineNumberWidth);
            }
            else
            {
                // 如果索引無效（超出文本範圍），繪製空白行號區域
                RenderEmptyLineNumber(Context.ViewPort.X, Context.ViewPort.Y + i, lineNumberDigits);
                
                // 繪製空白行
                RenderEmptyLine(Context.ViewPort.X + lineNumberWidth, Context.ViewPort.Y + i, Context.ViewPort.Width - lineNumberWidth);
            }
        }
        
        // 如果狀態欄可見，則繪製狀態欄
        if (IsStatusBarVisible)
        {
            RenderStatusBar();
        }

        // 繪製顯示框
        RenderFrame();

        // 設置光標位置，考慮偏移量和行號區域
        int cursorScreenX = Context.CursorX - Context.OffsetX + Context.ViewPort.X + lineNumberWidth;
        int cursorScreenY = Context.CursorY - Context.OffsetY + Context.ViewPort.Y;
        
        // 確保光標在可見區域內，且不在行號區域
        if (cursorScreenX >= Context.ViewPort.X + lineNumberWidth && 
            cursorScreenX < Context.ViewPort.X + Context.ViewPort.Width &&
            cursorScreenY >= Context.ViewPort.Y && 
            cursorScreenY < Context.ViewPort.Y + Context.ViewPort.Height)
        {
            _console.SetCursorPosition(cursorScreenX, cursorScreenY);
        }
        
        // 顯示游標
        _console.Write("\x1b[?25h");
    }
    
    /// <summary>
    /// 繪製行號
    /// </summary>
    private void RenderLineNumber(int x, int y, int lineNumber, int digits, bool isCurrentLine)
    {
        // 檢查 Y 座標是否在 ViewPort 範圍內
        if (y < Context.ViewPort.Y || y >= Context.ViewPort.Y + Context.ViewPort.Height)
        {
            return; // Y 座標超出範圍，不繪製
        }

        // 設置光標位置到行號區域的起始位置
        _console.SetCursorPosition(x, y);
        
        // 創建 StringBuilder 來構建行號字符串
        var sb = new StringBuilder();
        
        if (isCurrentLine)
        {
            // 當前行顯示空格
            for (int i = 0; i < digits; i++)
            {
                var coloredChar = new ColoredChar
                {
                    Char = ' ',
                    Foreground = ConsoleColor.White,
                    Background = ConsoleColor.DarkGray
                };
                sb.Append(coloredChar.ToAnsiString());
            }
        }
        else
        {
            // 格式化行號，靠右對齊
            string lineNumberStr = lineNumber.ToString().PadLeft(digits);
            
            // 添加行號
            foreach (char c in lineNumberStr)
            {
                var coloredChar = new ColoredChar
                {
                    Char = c,
                    Foreground = ConsoleColor.Yellow,
                    Background = ConsoleColor.DarkGray
                };
                sb.Append(coloredChar.ToAnsiString());
            }
        }
        
        // 添加一個空格作為間隔
        var spaceChar = new ColoredChar
        {
            Char = ' ',
            Foreground = ConsoleColor.White,
            Background = ConsoleColor.DarkGray
        };
        sb.Append(spaceChar.ToAnsiString());
        
        // 輸出構建好的字符串
        _console.Write(sb.ToString());
    }
    
    /// <summary>
    /// 繪製空白行號區域
    /// </summary>
    private void RenderEmptyLineNumber(int x, int y, int digits)
    {
        // 檢查 Y 座標是否在 ViewPort 範圍內
        if (y < Context.ViewPort.Y || y >= Context.ViewPort.Y + Context.ViewPort.Height)
        {
            return; // Y 座標超出範圍，不繪製
        }

        // 設置光標位置到行號區域的起始位置
        _console.SetCursorPosition(x, y);
        
        // 創建 StringBuilder 來構建空白行號字符串
        var sb = new StringBuilder();
        
        // 添加空白字符，寬度等於行號區域寬度
        for (int i = 0; i < digits + 1; i++)
        {
            var coloredChar = new ColoredChar
            {
                Char = ' ',
                Foreground = ConsoleColor.White,
                Background = ConsoleColor.DarkGray
            };
            sb.Append(coloredChar.ToAnsiString());
        }
        
        // 輸出構建好的字符串
        _console.Write(sb.ToString());
    }
    
    /// <summary>
    /// 繪製文本，考慮 ViewPort 和偏移量
    /// </summary>
    private void RenderText(int x, int y, ConsoleText text, int offset, ConsoleRectangle viewPort, int lineNumberWidth = 0)
    {
        // 檢查 Y 座標是否在 ViewPort 範圍內
        if (y < viewPort.Y || y >= viewPort.Y + viewPort.Height)
        {
            return; // Y 座標超出範圍，不繪製
        }

        // 設置光標位置到可見區域的起始位置
        _console.SetCursorPosition(x, y);
        
        // 計算可見區域的寬度，考慮行號區域
        int visibleWidth = viewPort.Width - lineNumberWidth;
        
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

    /// <summary>
    /// 繪製顯示框
    /// </summary>
    private void RenderFrame()
    {
        // 定義框架字符
        char topLeft = '┌';
        char topRight = '┐';
        char bottomLeft = '└';
        char bottomRight = '┘';
        char horizontal = '─';
        char vertical = '│';
        
        // 設置框架顏色
        ConsoleColor frameColor = ConsoleColor.Cyan;
        ConsoleColor backgroundColor = ConsoleColor.Black;
        
        // 計算框架的位置
        int frameX = Context.ViewPort.X - 1;
        int frameY = Context.ViewPort.Y - 1;
        int frameWidth = Context.ViewPort.Width + 2;
        int frameHeight = Context.ViewPort.Height + 2;
        
        // 繪製頂部邊框
        _console.SetCursorPosition(frameX, frameY);
        var sbTop = new StringBuilder();
        
        // 添加左上角
        var topLeftChar = new ColoredChar
        {
            Char = topLeft,
            Foreground = frameColor,
            Background = backgroundColor
        };
        sbTop.Append(topLeftChar.ToAnsiString());
        
        // 添加頂部水平線
        for (int i = 0; i < Context.ViewPort.Width; i++)
        {
            var horizontalChar = new ColoredChar
            {
                Char = horizontal,
                Foreground = frameColor,
                Background = backgroundColor
            };
            sbTop.Append(horizontalChar.ToAnsiString());
        }
        
        // 添加右上角
        var topRightChar = new ColoredChar
        {
            Char = topRight,
            Foreground = frameColor,
            Background = backgroundColor
        };
        sbTop.Append(topRightChar.ToAnsiString());
        
        _console.Write(sbTop.ToString());
        
        // 繪製左右邊框
        for (int i = 0; i < Context.ViewPort.Height; i++)
        {
            // 左邊框
            _console.SetCursorPosition(frameX, frameY + 1 + i);
            var leftVerticalChar = new ColoredChar
            {
                Char = vertical,
                Foreground = frameColor,
                Background = backgroundColor
            };
            _console.Write(leftVerticalChar.ToAnsiString());
            
            // 右邊框
            _console.SetCursorPosition(frameX + frameWidth - 1, frameY + 1 + i);
            var rightVerticalChar = new ColoredChar
            {
                Char = vertical,
                Foreground = frameColor,
                Background = backgroundColor
            };
            _console.Write(rightVerticalChar.ToAnsiString());
        }
        
        // 繪製底部邊框
        _console.SetCursorPosition(frameX, frameY + frameHeight - 1);
        var sbBottom = new StringBuilder();
        
        // 添加左下角
        var bottomLeftChar = new ColoredChar
        {
            Char = bottomLeft,
            Foreground = frameColor,
            Background = backgroundColor
        };
        sbBottom.Append(bottomLeftChar.ToAnsiString());
        
        // 添加底部水平線
        for (int i = 0; i < Context.ViewPort.Width; i++)
        {
            var horizontalChar = new ColoredChar
            {
                Char = horizontal,
                Foreground = frameColor,
                Background = backgroundColor
            };
            sbBottom.Append(horizontalChar.ToAnsiString());
        }
        
        // 添加右下角
        var bottomRightChar = new ColoredChar
        {
            Char = bottomRight,
            Foreground = frameColor,
            Background = backgroundColor
        };
        sbBottom.Append(bottomRightChar.ToAnsiString());
        
        _console.Write(sbBottom.ToString());
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