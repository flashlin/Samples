namespace VimSharpLib;

using System.Text;
using System.Collections.Generic;

public class VimEditor
{
    public IConsoleDevice Console { get; private set; }

    public bool IsRunning { get; set; } = true;
    public ConsoleContext Context { get; set; } = new();
    public IVimMode Mode { get; set; } = null!;

    // 添加剪貼簿緩衝區
    public List<ConsoleText> ClipboardBuffers { get; set; } = [];

    public VimEditor() : this(new ConsoleDevice())
    {
    }

    public VimEditor(IConsoleDevice console)
    {
        Console = console;
        Mode = new VimNormalMode { Instance = this };
        Initialize();
    }

    public void Initialize()
    {
        Context.SetViewPort(0, 0, Console.WindowWidth, Console.WindowHeight);
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
        // 初始化 screenBuffer，如果是 null
        if (screenBuffer == null)
        {
            screenBuffer = CreateScreenBuffer();
        }

        // 繪製內容區域
        RenderContentArea(screenBuffer, Context.GetLineNumberWidth());

        // 如果狀態欄可見，則繪製狀態欄到 screenBuffer
        if (Context.IsStatusBarVisible)
        {
            RenderStatusBar(screenBuffer);
        }

        // 繪製顯示框到 screenBuffer
        RenderFrame(screenBuffer);
    }

    public ColoredChar[,] CreateScreenBuffer()
    {
        int width = Console.WindowWidth;
        int height = Console.WindowHeight;
        var screenBuffer = new ColoredChar[width, height];
        // 初始化整個 screenBuffer
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                screenBuffer[x, y] = new ColoredChar(' ', ConsoleColor.White, ConsoleColor.Black);
            }
        }
        return screenBuffer;
    }

    public void RenderToConsole(ColoredChar[,] screenBuffer)
    {
        // 創建一個緩衝區用於收集所有輸出
        var outputBuffer = new StringBuilder();
        outputBuffer.Append($"\x1b[0;0H");
        // 隱藏游標 (符合 Rule 12)
        outputBuffer.Append("\x1b[?25l");

        RenderBufferToConsole(screenBuffer, outputBuffer);

        // 設置控制台游標位置
        outputBuffer.Append($"\x1b[{Context.CursorY+1};{Context.CursorX+1}H");
        
        // 顯示游標
        outputBuffer.Append("\x1b[?25h");
        // 一次性輸出所有內容到控制台
        Console.Write(outputBuffer.ToString());
    }

    /// <summary>
    /// 計算行號區域所需的位數
    /// </summary>
    private int CalculateLineNumberDigits()
    {
        // 計算文本總行數
        int totalLines = Context.Texts.Count;

        // 獲取游標在文本中的實際 Y 坐標
        int cursorTextY = GetActualTextY();

        // 計算行號區域寬度所需的位數
        if (Context.IsLineNumberVisible)
        {
            // 相對行號模式：計算相對行號的最大值（上下行數的最大值）
            int maxRelativeLineNumber = Math.Max(cursorTextY, totalLines - cursorTextY - 1);
            return maxRelativeLineNumber > 0 ? (int)Math.Log10(maxRelativeLineNumber) + 1 : 1;
        }
        else
        {
            // 絕對行號模式：計算總行數所需的位數
            return totalLines > 0 ? (int)Math.Log10(totalLines) + 1 : 1;
        }
    }

    /// <summary>
    /// 繪製內容區域（包括行號和文本）
    /// </summary>
    private void RenderContentArea(ColoredChar[,] screenBuffer, int lineNumberWidth)
    {
        // 獲取游標在文本中的實際 Y 坐標
        var cursorTextY = GetActualTextY();
        
        // 如果狀態欄可見，則減少一行用於顯示狀態欄
        int maxLines = Context.IsStatusBarVisible ? Context.ViewPort.Height - 1 : Context.ViewPort.Height;

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

                if (Context.IsLineNumberVisible)
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

                // 繪製行號到 screenBuffer
                RenderLineNumber(screenBuffer, Context.ViewPort.X, Context.ViewPort.Y + i, lineNumber, !isCurrentLine);

                // 繪製文本到 screenBuffer，考慮 ViewPort、偏移量和行號區域
                RenderText(screenBuffer, Context.ViewPort.X + lineNumberWidth, Context.ViewPort.Y + i, text, Context.OffsetX,
                    Context.ViewPort, lineNumberWidth);
            }
            else
            {
                // 如果索引無效（超出文本範圍），繪製空白行號區域到 screenBuffer
                RenderEmptyLineNumber(screenBuffer, Context.ViewPort.X, Context.ViewPort.Y + i, lineNumberWidth);

                // 繪製空白行到 screenBuffer
                RenderEmptyLine(screenBuffer, Context.ViewPort.X + lineNumberWidth, Context.ViewPort.Y + i,
                    Context.ViewPort.Width - lineNumberWidth);
            }
        }
    }

    /// <summary>
    /// 繪製狀態欄到 screenBuffer
    /// </summary>
    private void RenderStatusBar(ColoredChar[,] screenBuffer)
    {
        // 計算狀態欄的位置
        int statusBarY = Context.ViewPort.Y + Context.ViewPort.Height - 1;
        
        // 檢查 Y 座標是否在 screenBuffer 範圍內
        if (statusBarY < 0 || statusBarY >= screenBuffer.GetLength(1))
        {
            return; // Y 座標超出範圍，不繪製
        }

        // 獲取狀態欄文本
        string statusText = "";
        if (Context.StatusBar.Chars.Length == 0 || Context.StatusBar.Chars.All(c => c.Char == '\0'))
        {
            // 如果沒有設置狀態欄文本，則顯示默認信息
            string modeName = Mode.GetType().Name.Replace("Vim", "").Replace("Mode", "");
            statusText = $" {modeName} | Line: {Context.CursorY + 1} | Col: {Context.CursorX + 1} ";

            // 更新 StatusBar
            Context.StatusBar = new ConsoleText();
            Context.StatusBar.SetText(0, statusText);

            // 設置反色顯示
            for (int i = 0; i < Context.StatusBar.Chars.Length; i++)
            {
                if (Context.StatusBar.Chars[i].Char != '\0')
                {
                    Context.StatusBar.Chars[i] = new ColoredChar(Context.StatusBar.Chars[i].Char, ConsoleColor.Black, ConsoleColor.White);
                }
            }
        }
        else
        {
            // 使用已有的狀態欄文本
            statusText = new string(Context.StatusBar.Chars.Select(c => c.Char).ToArray());
        }

        // 確保狀態欄文本不超過視窗寬度
        if (statusText.Length > Context.ViewPort.Width)
        {
            statusText = statusText.Substring(0, Context.ViewPort.Width);
            
            // 更新 StatusBar 長度
            var newStatusBar = new ConsoleText();
            newStatusBar.SetText(0, statusText);
            
            // 設置反色顯示
            for (int i = 0; i < newStatusBar.Chars.Length; i++)
            {
                if (newStatusBar.Chars[i].Char != '\0')
                {
                    newStatusBar.Chars[i] = new ColoredChar(newStatusBar.Chars[i].Char, ConsoleColor.Black, ConsoleColor.White);
                }
            }
            
            Context.StatusBar = newStatusBar;
        }
        else if (statusText.Length < Context.ViewPort.Width)
        {
            // 如果狀態欄文本不夠長，則用空格填充
            statusText = statusText.PadRight(Context.ViewPort.Width);
            
            // 更新 StatusBar
            var newStatusBar = new ConsoleText();
            newStatusBar.SetText(0, statusText);
            
            // 設置反色顯示
            for (int i = 0; i < newStatusBar.Chars.Length; i++)
            {
                if (newStatusBar.Chars[i].Char != '\0')
                {
                    newStatusBar.Chars[i] = new ColoredChar(newStatusBar.Chars[i].Char, ConsoleColor.Black, ConsoleColor.White);
                }
            }
            
            Context.StatusBar = newStatusBar;
        }

        // 繪製狀態欄到 screenBuffer
        for (int i = 0; i < Context.StatusBar.Width; i++)
        {
            if (Context.ViewPort.X + i >= 0 && Context.ViewPort.X + i < screenBuffer.GetLength(0) && i < Context.StatusBar.Chars.Length)
            {
                screenBuffer[Context.ViewPort.X + i, statusBarY] = Context.StatusBar.Chars[i];
            }
        }
    }

    /// <summary>
    /// 將 screenBuffer 轉換為 ANSI 控制碼並輸出到 outputBuffer
    /// </summary>
    private void RenderBufferToConsole(ColoredChar[,] screenBuffer, StringBuilder outputBuffer)
    {
        // 輸出整個 screenBuffer 的內容
        for (int y = 0; y < screenBuffer.GetLength(1); y++)
        {
            // 直接輸出每一行的內容
            for (int x = 0; x < screenBuffer.GetLength(0); x++)
            {
                outputBuffer.Append(screenBuffer[x, y].ToAnsiString());
            }
        }
    }

    /// <summary>
    /// 繪製行號到 screenBuffer
    /// </summary>
    private void RenderLineNumber(ColoredChar[,] screenBuffer, int x, int y, int lineNumber, bool isRelativeLineNumber)
    {
        // 檢查 Y 座標是否在 ViewPort 範圍內
        if (y < Context.ViewPort.Y || y >= Context.ViewPort.Y + Context.ViewPort.Height)
        {
            return;
        }

        var digits = Context.Texts.Count.ToString().Length;
        var lineNumberChars = new ConsoleText();
        if (isRelativeLineNumber)
        {
            var foreground = ConsoleColor.White;
            var background = ConsoleColor.Blue;
            lineNumberChars.SetText(0, lineNumber.ToString().PadLeft(digits));
            lineNumberChars.SetColor(foreground, background);
            // 添加一個空格作為間隔
            screenBuffer[x + digits, y] = new ColoredChar(' ', foreground, background);
        }
        else
        {
            var foreground = ConsoleColor.Blue;
            var background = ConsoleColor.Black;
            lineNumberChars.SetText(0, lineNumber.ToString().PadRight(digits));
            lineNumberChars.SetColor(foreground, background);
            // 添加一個空格作為間隔
            screenBuffer[x + digits, y] = new ColoredChar(' ', foreground, background);
        }
        screenBuffer.SetText(x, y, lineNumberChars);
    }

    /// <summary>
    /// 繪製空白行號區域到 screenBuffer
    /// </summary>
    private void RenderEmptyLineNumber(ColoredChar[,] screenBuffer, int x, int y, int lineNumbderWidth)
    {
        // 檢查 Y 座標是否在 ViewPort 範圍內
        if (y < Context.ViewPort.Y || y >= Context.ViewPort.Y + Context.ViewPort.Height)
        {
            return; // Y 座標超出範圍，不繪製
        }

        // 添加空白字符，寬度等於行號區域寬度
        for (int i = 0; i < lineNumbderWidth; i++)
        {
            screenBuffer[x + i, y] = new ColoredChar(' ', ConsoleColor.White, ConsoleColor.Black);
        }
    }

    /// <summary>
    /// 繪製文本到 screenBuffer，考慮 ViewPort 和偏移量
    /// </summary>
    private void RenderText(ColoredChar[,] screenBuffer, int x, int y, ConsoleText text, int offset, ViewArea viewPort,
        int lineNumberWidth = 0)
    {
        // 檢查 Y 座標是否在 ViewPort 範圍內
        if (y < viewPort.Y || y >= viewPort.Y + viewPort.Height)
        {
            return; // Y 座標超出範圍，不繪製
        }

        // 計算可見區域的寬度，考慮行號區域
        int visibleWidth = viewPort.Width - lineNumberWidth;

        // 計算可見的起始和結束位置 (使用 text.Width 屬性)
        int startX = Math.Max(0, offset);
        int endX = Math.Min(text.Width, offset + visibleWidth);

        // 計算實際要繪製的字符數量
        int charsToDraw = endX - startX;

        // 如果有文本內容在可見範圍內
        if (startX < text.Width && charsToDraw > 0)
        {
            // 繪製文本內容
            for (int i = 0; i < charsToDraw; i++)
            {
                int textPos = startX + i;
                if (x + i >= 0 && x + i < screenBuffer.GetLength(0) && y >= 0 && y < screenBuffer.GetLength(1))
                {
                    var c = text.Chars[textPos];
                    if (c.Char == '\0')
                    {
                        // 如果是空字符，添加一個空格（黑底白字）
                        screenBuffer[x + i, y] = ColoredChar.Empty;
                    }
                    else
                    {
                        screenBuffer[x + i, y] = c;
                    }
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
                int pos = x + charsToDraw + i;
                if (pos >= 0 && pos < screenBuffer.GetLength(0) && y >= 0 && y < screenBuffer.GetLength(1))
                {
                    screenBuffer[pos, y] = ColoredChar.Empty;
                }
            }
        }
    }

    /// <summary>
    /// 繪製空白行到 screenBuffer
    /// </summary>
    private void RenderEmptyLine(ColoredChar[,] screenBuffer, int x, int y, int width)
    {
        // 檢查 Y 座標是否在 ViewPort 範圍內
        if (y < Context.ViewPort.Y || y >= Context.ViewPort.Y + Context.ViewPort.Height)
        {
            return; // Y 座標超出範圍，不繪製
        }

        // 填充空白字符
        for (int i = 0; i < width; i++)
        {
            if (x + i >= 0 && x + i < screenBuffer.GetLength(0) && y >= 0 && y < screenBuffer.GetLength(1))
            {
                screenBuffer[x + i, y] = ColoredChar.Empty;
            }
        }
    }

    /// <summary>
    /// 繪製顯示框到 screenBuffer
    /// </summary>
    private void RenderFrame(ColoredChar[,] screenBuffer)
    {
        // 定義框架字符
        char topLeft = '┌';
        char topRight = '┐';
        char bottomLeft = '└';
        char bottomRight = '┘';
        char horizontal = '─';
        char vertical = '│';

        // 框架顏色
        ConsoleColor frameColor = ConsoleColor.White;
        ConsoleColor backgroundColor = ConsoleColor.Black;

        // 計算框架位置和大小
        int frameX = Context.ViewPort.X - 1;
        int frameY = Context.ViewPort.Y - 1;
        int frameWidth = Context.ViewPort.Width + 2;
        int frameHeight = Context.ViewPort.Height + 2;

        // 計算有效的框架範圍（確保不超出 screenBuffer 邊界）
        int startX = Math.Max(0, frameX);
        int startY = Math.Max(0, frameY);
        int endX = Math.Min(screenBuffer.GetLength(0) - 1, frameX + frameWidth - 1);
        int endY = Math.Min(screenBuffer.GetLength(1) - 1, frameY + frameHeight - 1);

        // 繪製頂部邊框
        if (startY == frameY) // 頂部邊框在可見範圍內
        {
            // 繪製頂部水平線
            for (int x = startX; x <= endX; x++)
            {
                char c = horizontal;
                if (x == frameX) c = topLeft; // 左上角
                else if (x == frameX + frameWidth - 1) c = topRight; // 右上角
                
                screenBuffer[x, startY] = new ColoredChar(c, frameColor, backgroundColor);
            }
        }

        // 繪製底部邊框
        if (endY == frameY + frameHeight - 1) // 底部邊框在可見範圍內
        {
            // 繪製底部水平線
            for (int x = startX; x <= endX; x++)
            {
                char c = horizontal;
                if (x == frameX) c = bottomLeft; // 左下角
                else if (x == frameX + frameWidth - 1) c = bottomRight; // 右下角
                
                screenBuffer[x, endY] = new ColoredChar(c, frameColor, backgroundColor);
            }
        }

        // 繪製左右邊框
        for (int y = startY; y <= endY; y++)
        {
            // 左邊框
            if (startX == frameX)
            {
                // 跳過已經繪製的角落
                if (y != frameY && y != frameY + frameHeight - 1)
                {
                    screenBuffer[startX, y] = new ColoredChar(vertical, frameColor, backgroundColor);
                }
            }

            // 右邊框
            if (endX == frameX + frameWidth - 1)
            {
                // 跳過已經繪製的角落
                if (y != frameY && y != frameY + frameHeight - 1)
                {
                    screenBuffer[endX, y] = new ColoredChar(vertical, frameColor, backgroundColor);
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
                int lineNumberWidth = Context.IsLineNumberVisible ? Context.GetLineNumberWidth() : 0;
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
            int lineNumberWidth = Context.IsLineNumberVisible ? Context.GetLineNumberWidth() : 0;
            
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
        int lineNumberWidth = Context.GetLineNumberWidth();

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
        if (Context.IsLineNumberVisible && textX < lineNumberWidth)
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
        if (Context.IsStatusBarVisible && Context.CursorY == Context.Texts.Count - 1 &&
            Context.CursorY - Context.OffsetY >= effectiveViewPortHeight - 1)
        {
            // 如果游標在最後一行，且該行會被狀態欄覆蓋，則調整偏移量
            Context.OffsetY += 1;
        }
    }
    
    public int GetActualTextX()
    {
        return Context.CursorX - Context.ViewPort.X + Context.OffsetX - Context.GetLineNumberWidth();
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