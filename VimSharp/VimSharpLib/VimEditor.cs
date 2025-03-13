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

    private bool _isRelativeLineNumber = false;
    public bool IsRelativeLineNumber
    {
        get => _isRelativeLineNumber;
        set
        {
            // 如果值沒有變化，則不需要處理
            if (_isRelativeLineNumber == value)
                return;
                
            // 保存當前游標位置
            int originalCursorX = Context.CursorX;
            int originalCursorY = Context.CursorY;
            
            // 更新屬性值
            _isRelativeLineNumber = value;
            
            // 如果啟用了相對行號，則需要調整游標位置
            if (value)
            {
                // 計算相對行號區域的寬度
                int lineNumberWidth = CalculateLineNumberWidth();
                
                // 調整游標位置
                Context.CursorX = lineNumberWidth;
            }
            else
            {
                // 恢復原始游標位置
                Context.CursorX = originalCursorX;
            }
            
            // 確保游標Y位置不變
            Context.CursorY = originalCursorY;
        }
    }
    
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
            SetViewPort(0, 0, _console.WindowWidth, _console.WindowHeight);
        }
    }

    /// <summary>
    /// 設置視窗的矩形區域並調整游標位置
    /// </summary>
    /// <param name="x">視窗左上角的 X 座標</param>
    /// <param name="y">視窗左上角的 Y 座標</param>
    /// <param name="width">視窗的寬度</param>
    /// <param name="height">視窗的高度</param>
    public void SetViewPort(int x, int y, int width, int height)
    {
        Context.ViewPort = new ConsoleRectangle(x, y, width, height);
        Context.CursorY = y;
    }

    public void SetText(string text)
    {
        Context.Texts.Clear();
        // 將文本按照換行符分割並添加到 Texts 集合中
        string[] lines = text.Split(["\r\n", "\n"], StringSplitOptions.None);
        foreach (var line in lines)
        {
            Context.Texts.Add(new ConsoleText());
            Context.Texts[Context.Texts.Count - 1].SetText(0, line);
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
        
        // 計算行號區域寬度所需的位數
        int lineNumberDigits;
        
        if (IsRelativeLineNumber)
        {
            // 相對行號模式：計算相對行號的最大值（上下行數的最大值）
            int maxRelativeLineNumber = Math.Max(Context.CursorY, totalLines - Context.CursorY - 1);
            lineNumberDigits = maxRelativeLineNumber > 0 ? (int)Math.Log10(maxRelativeLineNumber) + 1 : 1;
        }
        else
        {
            // 絕對行號模式：計算總行數所需的位數
            lineNumberDigits = totalLines > 0 ? (int)Math.Log10(totalLines) + 1 : 1;
        }
        
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
                
                // 計算行號
                int lineNumber;
                bool isCurrentLine = (textIndex == Context.CursorY);
                
                if (IsRelativeLineNumber)
                {
                    // 相對行號模式
                    if (isCurrentLine)
                    {
                        // 當前行顯示絕對行號
                        lineNumber = textIndex + 1; // 顯示給用戶的行號從1開始
                    }
                    else
                    {
                        // 其他行顯示相對行號
                        lineNumber = Math.Abs(textIndex - Context.CursorY);
                    }
                }
                else
                {
                    // 絕對行號模式
                    lineNumber = textIndex + 1; // 顯示給用戶的行號從1開始
                }
                
                // 繪製行號
                RenderLineNumber(Context.ViewPort.X, Context.ViewPort.Y + i, lineNumber, lineNumberDigits, isCurrentLine, IsRelativeLineNumber);
                
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

        // 設置控制台游標位置
        _console.SetCursorPosition(Context.CursorX, Context.CursorY);
        // 顯示游標
        _console.Write("\x1b[?25h");
    }
    
    /// <summary>
    /// 繪製行號
    /// </summary>
    private void RenderLineNumber(int x, int y, int lineNumber, int digits, bool isCurrentLine, bool isRelativeLineNumber)
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
            // 當前行顯示絕對行號，使用不同顏色
            string lineNumberStr = lineNumber.ToString().PadLeft(digits);
            
            foreach (char c in lineNumberStr)
            {
                var coloredChar = new ColoredChar
                {
                    Char = c,
                    Foreground = ConsoleColor.White, // 當前行使用白色
                    Background = ConsoleColor.Black
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
                    Foreground = isRelativeLineNumber ? ConsoleColor.Yellow : ConsoleColor.Gray, // 相對行號使用黃色，絕對行號使用灰色
                    Background = ConsoleColor.Black
                };
                sb.Append(coloredChar.ToAnsiString());
            }
        }
        
        // 添加一個空格作為間隔
        var spaceChar = new ColoredChar
        {
            Char = ' ',
            Foreground = ConsoleColor.White,
            Background = ConsoleColor.Black
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
                Background = ConsoleColor.Black
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
    
    /// <summary>
    /// 計算相對行號區域的寬度
    /// </summary>
    /// <returns>相對行號區域的寬度</returns>
    public int CalculateLineNumberWidth()
    {
        if (!IsRelativeLineNumber)
        {
            return 0;
        }
        
        // 根據 .cursorrules 中的 RelativeLineNumerWidth Logic 規則
        // 返回固定值 2（1位數字+1位空格）
        return 2;
    }
    
    /// <summary>
    /// 檢查並調整游標位置和偏移量，確保游標在可見區域內
    /// </summary>
    public void AdjustCursorAndOffset()
    {
        // 計算行號區域寬度
        int lineNumberWidth = IsRelativeLineNumber ? CalculateLineNumberWidth() : 0;
        
        // 確保當前行存在
        if (Context.Texts.Count == 0)
        {
            Context.Texts.Add(new ConsoleText());
        }
        
        // 確保游標在文本範圍內
        Context.CursorY = Math.Min(Context.CursorY, Context.Texts.Count - 1);
        Context.CursorY = Math.Max(0, Context.CursorY);
        
        // 處理相對行號區域寬度
        if (IsRelativeLineNumber && Context.CursorX < lineNumberWidth)
        {
            Context.CursorX = lineNumberWidth;
        }
        
        // 確保游標水平位置在當前行文本範圍內
        var currentLine = Context.Texts[Context.CursorY];
        string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
        int textWidth = currentText.GetStringDisplayWidth();
        
        // 游標可以停在最後一個字符上或行號區域寬度處
        Context.CursorX = Math.Min(Context.CursorX, Math.Max(lineNumberWidth, textWidth));
        
        // 計算游標在屏幕上的位置
        int cursorScreenX = Context.CursorX - Context.OffsetX;
        int cursorScreenY = Context.CursorY - Context.OffsetY;
        
        // 計算可見區域的有效高度（考慮狀態欄）
        int effectiveViewPortHeight = Context.IsStatusBarVisible
            ? Context.ViewPort.Height - 1
            : Context.ViewPort.Height;
        
        // 檢查游標是否超出右邊界
        if (cursorScreenX >= Context.ViewPort.Width)
        {
            // 調整水平偏移量，使游標位於可見區域的右邊界
            Context.OffsetX = Context.CursorX - Context.ViewPort.Width + 1;
        }
        // 檢查游標是否超出左邊界
        else if (cursorScreenX < 0)
        {
            // 調整水平偏移量，使游標位於可見區域的左邊界
            Context.OffsetX = Context.CursorX;
        }
        
        // 檢查游標是否超出下邊界
        if (cursorScreenY >= effectiveViewPortHeight)
        {
            // 調整垂直偏移量，使游標位於可見區域的下邊界
            Context.OffsetY = Context.CursorY - effectiveViewPortHeight + 1;
        }
        // 檢查游標是否超出上邊界
        else if (cursorScreenY < 0)
        {
            // 調整垂直偏移量，使游標位於可見區域的上邊界
            Context.OffsetY = Context.CursorY;
        }
        
        // 處理狀態欄顯示
        if (Context.IsStatusBarVisible && Context.CursorY == Context.Texts.Count - 1 && 
            Context.CursorY - Context.OffsetY >= effectiveViewPortHeight)
        {
            // 如果游標在最後一行，且該行會被狀態欄覆蓋，則調整偏移量
            Context.OffsetY += 1;
        }
    }
}