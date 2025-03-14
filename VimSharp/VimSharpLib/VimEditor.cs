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
            if (_isRelativeLineNumber == value)
            {
                return;
            }

            _isRelativeLineNumber = value;
            var lineNumberWidth = CalculateLineNumberWidth();
            if (value)
            {
                Context.CursorX += lineNumberWidth;
            }
            else
            {
                Context.CursorX -= lineNumberWidth;
            }
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
        Mode = new VimNormalMode { Instance = this };
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
        Context.CursorX = x;
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

    public void Render(ColoredChar[,]? screenBuffer = null)
    {
        // 檢查是否直接輸出到控制台還是繪製到 screenBuffer
        bool directOutput = screenBuffer == null;
        
        // 創建一個緩衝區用於收集所有輸出
        var outputBuffer = new StringBuilder();

        // 如果是直接輸出，隱藏游標
        if (directOutput)
        {
            outputBuffer.Append("\x1b[?25l");
        }

        // 計算文本總行數
        int totalLines = Context.Texts.Count;

        // 獲取游標在文本中的實際 Y 坐標
        int cursorTextY = GetActualTextY();

        // 計算行號區域寬度所需的位數 (符合 RelativeLineNumerWidth Logic)
        int lineNumberDigits;

        if (IsRelativeLineNumber)
        {
            // 相對行號模式：計算相對行號的最大值（上下行數的最大值）
            int maxRelativeLineNumber = Math.Max(cursorTextY, totalLines - cursorTextY - 1);
            lineNumberDigits = maxRelativeLineNumber > 0 ? (int)Math.Log10(maxRelativeLineNumber) + 1 : 1;
        }
        else
        {
            // 絕對行號模式：計算總行數所需的位數
            lineNumberDigits = totalLines > 0 ? (int)Math.Log10(totalLines) + 1 : 1;
        }

        // 行號區域寬度 = 位數 + 1 (用於間隔)
        int lineNumberWidth = lineNumberDigits + 1;

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
                bool isCurrentLine = (textIndex == cursorTextY);

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
                        lineNumber = Math.Abs(textIndex - cursorTextY);
                    }
                }
                else
                {
                    // 絕對行號模式
                    lineNumber = textIndex + 1; // 顯示給用戶的行號從1開始
                }

                // 繪製行號，將輸出添加到緩衝區或 screenBuffer
                RenderLineNumber(outputBuffer, Context.ViewPort.X, Context.ViewPort.Y + i, lineNumber, lineNumberDigits,
                    isCurrentLine, IsRelativeLineNumber, screenBuffer);

                // 直接繪製文本，考慮 ViewPort、偏移量和行號區域，將輸出添加到緩衝區或 screenBuffer
                RenderText(outputBuffer, Context.ViewPort.X + lineNumberWidth, Context.ViewPort.Y + i, text, Context.OffsetX,
                    Context.ViewPort, lineNumberWidth, screenBuffer);
            }
            else
            {
                // 如果索引無效（超出文本範圍），繪製空白行號區域，將輸出添加到緩衝區或 screenBuffer
                RenderEmptyLineNumber(outputBuffer, Context.ViewPort.X, Context.ViewPort.Y + i, lineNumberDigits, screenBuffer);

                // 繪製空白行，將輸出添加到緩衝區或 screenBuffer
                RenderEmptyLine(outputBuffer, Context.ViewPort.X + lineNumberWidth, Context.ViewPort.Y + i,
                    Context.ViewPort.Width - lineNumberWidth, screenBuffer);
            }
        }

        // 如果狀態欄可見，則繪製狀態欄，將輸出添加到緩衝區或 screenBuffer
        if (IsStatusBarVisible)
        {
            RenderStatusBar(outputBuffer, screenBuffer);
        }

        // 繪製顯示框，輸出添加到緩衝區或 screenBuffer
        RenderFrame(outputBuffer, screenBuffer);

        // 如果是直接輸出到控制台
        if (directOutput)
        {
            // 設置控制台游標位置
            outputBuffer.Append($"\x1b[{Context.CursorY + 1};{Context.CursorX + 1}H");
            
            // 顯示游標
            outputBuffer.Append("\x1b[?25h");

            // 一次性輸出所有內容到控制台
            _console.Write(outputBuffer.ToString());
        }
    }

    /// <summary>
    /// 繪製行號，輸出添加到緩衝區或 screenBuffer
    /// </summary>
    private void RenderLineNumber(StringBuilder buffer, int x, int y, int lineNumber, int digits, bool isCurrentLine,
        bool isRelativeLineNumber, ColoredChar[,]? screenBuffer = null)
    {
        // 檢查 Y 座標是否在 ViewPort 範圍內
        if (y < Context.ViewPort.Y || y >= Context.ViewPort.Y + Context.ViewPort.Height)
        {
            return; // Y 座標超出範圍，不繪製
        }

        // 設置光標位置到行號區域的起始位置
        buffer.Append($"\x1b[{y + 1};{x + 1}H");

        // 創建臨時緩衝區來構建行號字符串
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

            foreach (char c in lineNumberStr)
            {
                var coloredChar = new ColoredChar
                {
                    Char = c,
                    Foreground = ConsoleColor.DarkGray, // 非當前行使用灰色
                    Background = ConsoleColor.Black
                };
                sb.Append(coloredChar.ToAnsiString());
            }
        }

        // 添加一個空白字符作為分隔符
        var spaceChar = new ColoredChar
        {
            Char = ' ',
            Foreground = ConsoleColor.DarkGray,
            Background = ConsoleColor.Black
        };
        sb.Append(spaceChar.ToAnsiString());

        // 將行號字符串添加到主緩衝區
        buffer.Append(sb.ToString());

        // 如果 screenBuffer 存在，將內容添加到 screenBuffer
        if (screenBuffer != null)
        {
            // 計算行號文本
            string lineNumberStr = lineNumber.ToString().PadLeft(digits);
            
            // 在 screenBuffer 中繪製行號
            for (int i = 0; i < digits; i++)
            {
                if (y >= 0 && y < screenBuffer.GetLength(0) && x + i >= 0 && x + i < screenBuffer.GetLength(1))
                {
                    var coloredChar = new ColoredChar
                    {
                        Char = i < lineNumberStr.Length ? lineNumberStr[i] : ' ',
                        Foreground = isCurrentLine ? ConsoleColor.White : ConsoleColor.DarkGray,
                        Background = ConsoleColor.Black
                    };
                    screenBuffer[y, x + i] = coloredChar;
                }
            }
            
            // 添加分隔符
            if (y >= 0 && y < screenBuffer.GetLength(0) && x + digits >= 0 && x + digits < screenBuffer.GetLength(1))
            {
                screenBuffer[y, x + digits] = new ColoredChar(' ', ConsoleColor.DarkGray, ConsoleColor.Black);
            }
        }
    }

    /// <summary>
    /// 繪製空白行號區域，輸出添加到緩衝區或 screenBuffer
    /// </summary>
    private void RenderEmptyLineNumber(StringBuilder buffer, int x, int y, int digits, ColoredChar[,]? screenBuffer = null)
    {
        // 檢查 Y 座標是否在 ViewPort 範圍內
        if (y < Context.ViewPort.Y || y >= Context.ViewPort.Y + Context.ViewPort.Height)
        {
            return; // Y 座標超出範圍，不繪製
        }

        // 設置光標位置到行號區域的起始位置
        buffer.Append($"\x1b[{y + 1};{x + 1}H");

        // 創建臨時緩衝區來構建空白行號區域
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

        // 將空白行號字符串添加到主緩衝區
        buffer.Append(sb.ToString());

        // 如果 screenBuffer 存在，將內容添加到 screenBuffer
        if (screenBuffer != null)
        {
            // 在 screenBuffer 中繪製空白行號區域
            for (int i = 0; i < digits + 1; i++)
            {
                if (y >= 0 && y < screenBuffer.GetLength(0) && x + i >= 0 && x + i < screenBuffer.GetLength(1))
                {
                    var coloredChar = new ColoredChar
                    {
                        Char = ' ',
                        Foreground = ConsoleColor.White,
                        Background = ConsoleColor.Black
                    };
                    screenBuffer[y, x + i] = coloredChar;
                }
            }
        }
    }

    /// <summary>
    /// 繪製文本，考慮 ViewPort 和偏移量，輸出添加到緩衝區或 screenBuffer
    /// </summary>
    private void RenderText(StringBuilder buffer, int x, int y, ConsoleText text, int offset, ConsoleRectangle viewPort,
        int lineNumberWidth = 0, ColoredChar[,]? screenBuffer = null)
    {
        // 檢查 Y 座標是否在 ViewPort 範圍內
        if (y < viewPort.Y || y >= viewPort.Y + viewPort.Height)
        {
            return; // Y 座標超出範圍，不繪製
        }

        // 設置光標位置到可見區域的起始位置
        buffer.Append($"\x1b[{y + 1};{x + 1}H");

        // 計算可見區域的寬度，考慮行號區域
        int visibleWidth = viewPort.Width - lineNumberWidth;

        // 創建臨時緩衝區來構建輸出字符串
        var sb = new StringBuilder();

        // 計算可見的起始和結束位置 (使用 text.Width 屬性)
        int startX = Math.Max(0, offset);
        int endX = Math.Min(text.Width, offset + visibleWidth);

        // 計算實際要繪製的字符數量
        int charsToDraw = endX - startX;

        // 如果有文本內容在可見範圍內
        if (startX < text.Width && charsToDraw > 0)
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

        // 將文本字符串添加到主緩衝區
        buffer.Append(sb.ToString());

        // 如果 screenBuffer 存在，將內容添加到 screenBuffer
        if (screenBuffer != null)
        {
            // 計算可見的文本區域
            int displayStartX = Math.Max(0, offset);
            int displayEndX = Math.Min(text.Width, offset + visibleWidth);
            int charsToDisplay = displayEndX - displayStartX;
            
            // 在 screenBuffer 中繪製文本內容
            if (displayStartX < text.Width && charsToDisplay > 0)
            {
                for (int i = 0; i < charsToDisplay; i++)
                {
                    if (y >= 0 && y < screenBuffer.GetLength(0) && x + i >= 0 && x + i < screenBuffer.GetLength(1))
                    {
                        int textIndex = displayStartX + i;
                        var c = text.Chars[textIndex];
                        if (c.Char == '\0')
                        {
                            // 如果是空字符，添加一個空格（黑底白字）
                            screenBuffer[y, x + i] = ColoredChar.Empty;
                        }
                        else
                        {
                            screenBuffer[y, x + i] = c;
                        }
                    }
                }
            }
            
            // 填充剩餘空白
            for (int i = charsToDisplay; i < visibleWidth && i < screenBuffer.GetLength(1) - x; i++)
            {
                if (y >= 0 && y < screenBuffer.GetLength(0) && x + i >= 0 && x + i < screenBuffer.GetLength(1))
                {
                    screenBuffer[y, x + i] = ColoredChar.Empty;
                }
            }
        }
    }

    /// <summary>
    /// 繪製空白行，輸出添加到緩衝區或 screenBuffer
    /// </summary>
    private void RenderEmptyLine(StringBuilder buffer, int x, int y, int width, ColoredChar[,]? screenBuffer = null)
    {
        // 檢查 Y 座標是否在控制台範圍內
        if (y < 0 || y >= _console.WindowHeight)
        {
            return; // Y 座標超出範圍，不繪製
        }

        // 設置光標位置
        buffer.Append($"\x1b[{y + 1};{x + 1}H");

        // 創建臨時緩衝區來構建輸出字符串
        var sb = new StringBuilder();

        // 填充空白字符
        for (int i = 0; i < width; i++)
        {
            sb.Append(ColoredChar.Empty.ToAnsiString());
        }

        // 將空白行字符串添加到主緩衝區
        buffer.Append(sb.ToString());

        // 如果 screenBuffer 存在，將內容添加到 screenBuffer
        if (screenBuffer != null)
        {
            // 在 screenBuffer 中繪製空白行
            for (int i = 0; i < width; i++)
            {
                if (y >= 0 && y < screenBuffer.GetLength(0) && x + i >= 0 && x + i < screenBuffer.GetLength(1))
                {
                    screenBuffer[y, x + i] = ColoredChar.Empty;
                }
            }
        }
    }

    /// <summary>
    /// 繪製狀態欄，輸出添加到緩衝區或 screenBuffer
    /// </summary>
    private void RenderStatusBar(StringBuilder buffer, ColoredChar[,]? screenBuffer = null)
    {
        // 計算狀態欄的位置
        int statusBarY = Context.ViewPort.Y + Context.ViewPort.Height - 1;

        // 設置光標位置到狀態欄
        buffer.Append($"\x1b[{statusBarY + 1};{Context.ViewPort.X + 1}H");

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

        // 創建臨時緩衝區來構建狀態欄字符串
        var sb = new StringBuilder();

        // 繪製狀態欄
        for (int i = 0; i < statusBarText.Width; i++)
        {
            sb.Append(statusBarText.Chars[i].ToAnsiString());
        }

        // 將狀態欄字符串添加到主緩衝區
        buffer.Append(sb.ToString());

        // 如果 screenBuffer 存在，將內容添加到 screenBuffer
        if (screenBuffer != null)
        {
            // 在 screenBuffer 中繪製狀態欄
            for (int i = 0; i < statusBarText.Width; i++)
            {
                if (statusBarY >= 0 && statusBarY < screenBuffer.GetLength(0) && 
                    Context.ViewPort.X + i >= 0 && Context.ViewPort.X + i < screenBuffer.GetLength(1))
                {
                    screenBuffer[statusBarY, Context.ViewPort.X + i] = statusBarText.Chars[i];
                }
            }
        }
    }

    /// <summary>
    /// 繪製顯示框，輸出添加到緩衝區或 screenBuffer
    /// </summary>
    private void RenderFrame(StringBuilder buffer, ColoredChar[,]? screenBuffer = null)
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
        buffer.Append($"\x1b[{frameY + 1};{frameX + 1}H");
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

        buffer.Append(sbTop.ToString());

        // 繪製側邊框
        for (int i = 0; i < Context.ViewPort.Height; i++)
        {
            int y = Context.ViewPort.Y + i;

            // 左側邊框
            buffer.Append($"\x1b[{y + 1};{frameX + 1}H");
            var leftChar = new ColoredChar
            {
                Char = vertical,
                Foreground = frameColor,
                Background = backgroundColor
            };
            buffer.Append(leftChar.ToAnsiString());

            // 右側邊框
            buffer.Append($"\x1b[{y + 1};{frameX + frameWidth}H");
            var rightChar = new ColoredChar
            {
                Char = vertical,
                Foreground = frameColor,
                Background = backgroundColor
            };
            buffer.Append(rightChar.ToAnsiString());
        }

        // 繪製底部邊框
        buffer.Append($"\x1b[{frameY + frameHeight};{frameX + 1}H");
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

        buffer.Append(sbBottom.ToString());

        // 如果 screenBuffer 存在，將內容添加到 screenBuffer
        if (screenBuffer != null)
        {
            // 在 screenBuffer 中繪製框架
            // 檢查框架是否在有效範圍內
            // 繪製頂部邊框
            if (frameY >= 0 && frameY < screenBuffer.GetLength(0))
            {
                for (int i = 0; i < frameWidth; i++)
                {
                    if (frameX + i >= 0 && frameX + i < screenBuffer.GetLength(1))
                    {
                        char c = i == 0 ? topLeft : (i == frameWidth - 1 ? topRight : horizontal);
                        screenBuffer[frameY, frameX + i] = new ColoredChar(c, frameColor, backgroundColor);
                    }
                }
            }

            // 繪製底部邊框
            if (frameY + frameHeight - 1 >= 0 && frameY + frameHeight - 1 < screenBuffer.GetLength(0))
            {
                for (int i = 0; i < frameWidth; i++)
                {
                    if (frameX + i >= 0 && frameX + i < screenBuffer.GetLength(1))
                    {
                        char c = i == 0 ? bottomLeft : (i == frameWidth - 1 ? bottomRight : horizontal);
                        screenBuffer[frameY + frameHeight - 1, frameX + i] = new ColoredChar(c, frameColor, backgroundColor);
                    }
                }
            }

            // 繪製左右邊框
            for (int i = 1; i < frameHeight - 1; i++)
            {
                if (frameY + i >= 0 && frameY + i < screenBuffer.GetLength(0))
                {
                    // 左側邊框
                    if (frameX >= 0 && frameX < screenBuffer.GetLength(1))
                    {
                        screenBuffer[frameY + i, frameX] = new ColoredChar(vertical, frameColor, backgroundColor);
                    }

                    // 右側邊框
                    if (frameX + frameWidth - 1 >= 0 && frameX + frameWidth - 1 < screenBuffer.GetLength(1))
                    {
                        screenBuffer[frameY + i, frameX + frameWidth - 1] = new ColoredChar(vertical, frameColor, backgroundColor);
                    }
                }
            }
        }
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
        Context.OffsetX += deltaX;
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
        return (int)Math.Floor(Math.Log10(Context.Texts.Count)) + 2;
    }

    public void MoveCursorRightN(int n)
    {
        for (int i = 0; i < n; i++)
        {
            MoveCursorRight();
        }
    }

    /// <summary>
    /// 處理向右移動時遇到的 '\0' 字符
    /// </summary>
    /// <param name="line">要處理的文本行</param>
    /// <param name="textX">起始 X 坐標</param>
    /// <returns>調整後的 X 坐標</returns>
    private int HandleNullCharForRightMovement(ConsoleText line, int textX)
    {
        // 確保 textX 在有效範圍內
        if (textX >= 0 && textX < line.Width)
        {
            // 檢查是否遇到 '\0' 字符，如果是則繼續向右移動直到遇到非 '\0' 字符
            while (textX < line.Width && line.Chars[textX].Char == '\0')
            {
                textX++;
            }
            
            // 如果已經到達行尾，則返回原始值
            if (textX >= line.Width)
            {
                return -1; // 表示已到達行尾
            }
        }
        else
        {
            return -1; // 表示索引無效
        }
        
        return textX;
    }

    public void MoveCursorRight()
    {
        var textX = GetActualTextX() + 1;
        var currentLine = GetCurrentLine();
        if (textX >= currentLine.Width)
        {
            return;
        }
        
        // 處理 '\0' 字符
        int adjustedTextX = HandleNullCharForRightMovement(currentLine, textX);
        if (adjustedTextX == -1)
        {
            return;
        }
        
        // 計算需要移動的實際步數
        int stepsToMove = adjustedTextX - GetActualTextX();
        var targetCursorX = Context.CursorX + stepsToMove;
        var remainingX = currentLine.Width - adjustedTextX;
        
        if (targetCursorX > Context.ViewPort.X + Context.ViewPort.Width - 1)
        {
            Scroll(remainingX, 0);
        }
        else
        {
            Context.CursorX = targetCursorX;
        }
    }

    /// <summary>
    /// 處理 '\0' 字符並調整游標位置
    /// </summary>
    /// <param name="line">要處理的文本行</param>
    /// <param name="actualTextX">實際文本 X 坐標</param>
    private void HandleNullCharAndAdjustCursor(ConsoleText line, int actualTextX)
    {
        // 確保 actualTextX 在有效範圍內
        if (actualTextX >= 0 && actualTextX < line.Width)
        {
            // 檢查目標位置是否為 '\0'，如果是則向左移動直到找到非 '\0' 字符
            if (line.Chars[actualTextX].Char == '\0')
            {
                int newTextX = actualTextX;
                
                // 向左移動直到找到非 '\0' 字符
                while (newTextX > 0 && line.Chars[newTextX].Char == '\0')
                {
                    newTextX--;
                }
                
                // 調整游標 X 位置
                int lineNumberWidth = IsRelativeLineNumber ? CalculateLineNumberWidth() : 0;
                Context.CursorX = Context.ViewPort.X + newTextX + lineNumberWidth - Context.OffsetX;
            }
        }
    }
    
    /// <summary>
    /// 調整游標位置以適應行寬度
    /// </summary>
    /// <param name="line">要處理的文本行</param>
    /// <param name="actualTextX">實際文本 X 坐標</param>
    private void AdjustCursorForLineWidth(ConsoleText line, int actualTextX)
    {
        // 如果行比較短，需要調整游標 X 位置
        if (actualTextX >= line.Width)
        {
            // 計算行的最後一個位置
            int lineNumberWidth = IsRelativeLineNumber ? CalculateLineNumberWidth() : 0;
            
            // 如果行是空的，將游標設置在行號後
            if (line.Width == 0)
            {
                Context.CursorX = Context.ViewPort.X + lineNumberWidth;
            }
            else
            {
                // 否則設置到行的末尾
                var lineDisplayWidth = line.Width + lineNumberWidth;
                Context.CursorX = Context.ViewPort.X + Math.Min(lineDisplayWidth - 1, Context.ViewPort.Width - 1);
            }
        }
    }

    /// <summary>
    /// 處理垂直滾動並更新游標 Y 位置
    /// </summary>
    /// <param name="targetCursorY">目標游標 Y 位置</param>
    /// <param name="scrollDirection">滾動方向，1 表示向下，-1 表示向上</param>
    private void HandleVerticalScrollAndUpdateCursor(int targetCursorY, int scrollDirection)
    {
        bool needScroll = false;
        
        if (scrollDirection > 0)
        {
            // 向下滾動
            needScroll = targetCursorY > Context.ViewPort.Y + Context.ViewPort.Height - 1;
        }
        else
        {
            // 向上滾動
            needScroll = targetCursorY < Context.ViewPort.Y;
        }
        
        if (needScroll)
        {
            // 需要滾動
            Scroll(0, scrollDirection);
        }
        else
        {
            // 不需要滾動，直接更新游標 Y 位置
            Context.CursorY = targetCursorY;
        }
    }

    public void MoveCursorDown()
    {
        // 檢查是否已經到達最後一行
        var textY = GetActualTextY() + 1;
        if (textY >= Context.Texts.Count)
        {
            return;
        }
        
        // 計算游標目標位置
        var targetCursorY = Context.CursorY + 1;
        
        // 獲取當前行和下一行
        var currentLine = GetCurrentLine();
        var nextLine = Context.Texts[textY];
        
        // 保存當前游標 X 位置
        var currentX = Context.CursorX;
        
        // 檢查下一行是否存在
        if (nextLine != null)
        {
            // 獲取文本實際 X 坐標
            var actualTextX = GetActualTextX();
            
            // 處理 '\0' 字符並調整游標位置
            HandleNullCharAndAdjustCursor(nextLine, actualTextX);
            
            // 調整游標位置以適應行寬度
            AdjustCursorForLineWidth(nextLine, actualTextX);
            
            // 處理垂直滾動並更新游標 Y 位置
            HandleVerticalScrollAndUpdateCursor(targetCursorY, 1);
        }
    }

    public void MoveCursorUp()
    {
        var textY = GetActualTextY() - 1;
        if (textY < 0)
        {
            return;
        }
        
        // 獲取上一行
        var prevLine = Context.Texts[textY];
        
        // 獲取文本實際 X 坐標
        var actualTextX = GetActualTextX();
        
        // 處理 '\0' 字符並調整游標位置
        HandleNullCharAndAdjustCursor(prevLine, actualTextX);
        
        // 調整游標位置以適應行寬度
        AdjustCursorForLineWidth(prevLine, actualTextX);
        
        // 處理垂直滾動並更新游標 Y 位置
        var targetCursorY = Context.CursorY - 1;
        HandleVerticalScrollAndUpdateCursor(targetCursorY, -1);
    }

    /// <summary>
    /// 檢查並調整游標位置和偏移量，確保游標在可見區域內
    /// </summary>
    public void AdjustCursorPositionAndOffset(int textX, int textY)
    {
        // 計算行號區域寬度
        int lineNumberWidth = CalculateLineNumberWidth();

        // 確保當前行存在
        if (Context.Texts.Count == 0)
        {
            Context.Texts.Add(new ConsoleText());
        }

        // 檢查 x, y 是否在 ViewPort 範圍內
        int maxX = Context.ViewPort.Width - 1;
        int maxY = Context.ViewPort.Height - 1;

        if (textX > maxX)
        {
            // 一般情況下，如果 x 超出了視口的最大寬度，設置適當的水平偏移量
            Context.OffsetX = textX - maxX;
            textX = maxX;
        }

        // 調整 x, y 以確保它們在有效範圍內
        textX = Math.Max(lineNumberWidth, Math.Min(textX, maxX));
        textY = Math.Max(0, Math.Min(textY, maxY));

        // 確保游標在文本範圍內
        Context.CursorY = Math.Min(textY, Context.Texts.Count - 1);
        Context.CursorY = Math.Max(0, Context.CursorY);

        // 處理相對行號區域寬度
        if (IsRelativeLineNumber && textX < lineNumberWidth)
        {
            textX = lineNumberWidth;
        }

        // 確保游標水平位置在當前行文本範圍內
        var currentLine = Context.Texts[Context.CursorY];
        string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
        int textWidth = currentText.GetStringDisplayWidth();

        // 游標可以停在最後一個字符上或行號區域寬度處
        Context.CursorX = Math.Min(textX, Math.Max(lineNumberWidth, textWidth));

        // 計算游標在屏幕上的位置
        int cursorScreenX = Context.CursorX - Context.OffsetX;
        int cursorScreenY = Context.CursorY - Context.OffsetY;

        // 計算可見區域的有效高度（考慮狀態欄）
        int effectiveViewPortHeight = Context.ViewPort.Height;

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
        if (IsStatusBarVisible && Context.CursorY == Context.Texts.Count - 1 &&
            Context.CursorY - Context.OffsetY >= effectiveViewPortHeight - 1)
        {
            // 如果游標在最後一行，且該行會被狀態欄覆蓋，則調整偏移量
            Context.OffsetY += 1;
        }
    }
    
    public int GetActualTextX()
    {
        return Context.CursorX - Context.ViewPort.X + Context.OffsetX - CalculateLineNumberWidth();
    }

    public int GetActualTextY()
    {
        return Context.CursorY - Context.ViewPort.Y + Context.OffsetY;
    }

    public ConsoleText GetCurrentLine()
    {
        return Context.Texts[GetActualTextY()];
    }
}