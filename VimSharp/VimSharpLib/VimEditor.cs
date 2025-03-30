namespace VimSharpLib;

using System.Text;
using System.Collections.Generic;
using System.Linq;

public class VimEditor
{
    public IConsoleDevice Console { get; private set; }

    public bool IsRunning { get; set; } = true;
    public ConsoleContext Context { get; set; } = new();
    public IVimMode Mode { get; set; } = null!;

    // 添加剪貼簿緩衝區
    public List<ConsoleText> ClipboardBuffers { get; set; } = [];

    public VimEditor(IConsoleDevice console)
    {
        Console = console;
        Mode = new VimNormalMode(this);
        Initialize();
    }

    private void Initialize()
    {
        Mode = new VimNormalMode(this);
        Context.SetViewPort(0, 0, Console.WindowWidth, Console.WindowHeight);
    }

    public void OpenText(string text)
    {
        Context.Texts.Clear();
        var lines = SplitText(text);
        foreach (var line in lines)
        {
            var consoleText = new ConsoleText();
            consoleText.SetText(0, line);
            Context.Texts.Add(consoleText);
        }
        Init();
    }

    private IEnumerable<string> SplitText(string text)
    {
        int start = 0;
        for (int i = 0; i < text.Length; i++)
        {
            if (i + 1 < text.Length && text[i] == '\r' && text[i + 1] == '\n')
            {
                yield return text[start..i] + "\n";
                start = i + 2;
                i++;
            }
            else if (text[i] == '\n')
            {
                yield return text[start..i] + "\n";
                start = i + 1;
            }
        }
        if (start < text.Length)
        {
            yield return text[start..];
        }
    }

    public void Init()
    {
        Context.CursorX = Context.ViewPort.X + Context.GetLineNumberWidth();
        Context.CursorY = Context.ViewPort.Y;
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
        if (screenBuffer == null)
        {
            screenBuffer = CreateScreenBuffer();
        }

        // 計算行號寬度
        int lineNumberWidth = Context.GetLineNumberWidth();
        
        // 獲取游標在文本中的實際位置
        int cursorTextY = GetActualTextY();
        
        // 繪製內容區域（包括行號和文本）
        RenderContentArea(screenBuffer, lineNumberWidth, cursorTextY);
        
        // 繪製狀態欄
        RenderStatusBar(screenBuffer);

        RenderFrame(screenBuffer);
        
        Mode.Render(screenBuffer);
        WriteToConsole(screenBuffer);
    }

    /// <summary>
    /// 繪製內容區域（包括行號和文本）
    /// </summary>
    private void RenderContentArea(ColoredChar[,] screenBuffer, int lineNumberWidth, int cursorTextY)
    {
        int bufferWidth = screenBuffer.GetLength(1);
        
        // 如果狀態欄可見，則減少一行用於顯示狀態欄
        int maxLines = Context.IsStatusBarVisible ? Context.ViewPort.Height - 1 : Context.ViewPort.Height;

        // 只繪製可見區域內的行
        for (var viewY = 0; viewY < maxLines && Context.ViewPort.Y + viewY < screenBuffer.GetLength(0); viewY++)
        {
            // 計算實際要繪製的文本行索引
            int textIndex = Context.OffsetY + viewY;

            // 繪製行號
            if (Context.IsLineNumberVisible)
            {
                RenderLineNumberForRow(screenBuffer, viewY, textIndex, cursorTextY, lineNumberWidth, bufferWidth);
            }

            // 繪製文字內容
            if (textIndex < Context.Texts.Count)
            {
                var textLine = Context.Texts[textIndex];
                RenderTextForRow(screenBuffer, viewY, textLine, lineNumberWidth);
            }
            else
            {
                // 填充空行
                RenderEmptyLine(screenBuffer, Context.ViewPort.X + lineNumberWidth, Context.ViewPort.Y + viewY,
                    Context.ViewPort.Width - lineNumberWidth);
            }
        }
    }
    
    /// <summary>
    /// 為指定行繪製行號
    /// </summary>
    private void RenderLineNumberForRow(ColoredChar[,] screenBuffer, int viewY, int textIndex, int cursorTextY, 
        int lineNumberWidth, int bufferWidth)
    {
        string lineNumber;
        if (textIndex == cursorTextY)
        {
            lineNumber = (textIndex + 1).ToString(); // 當前行顯示絕對行號
        }
        else
        {
            lineNumber = Math.Abs(textIndex - cursorTextY).ToString(); // 其他行顯示相對行號
        }

        // 確保行號不超過分配的空間
        for (int j = 0; j < lineNumberWidth; j++)
        {
            if (Context.ViewPort.X + j < bufferWidth)
            {
                var color = (textIndex == cursorTextY) ? ConsoleColor.Yellow : ConsoleColor.DarkGray;
                char charToRender = (j < lineNumber.Length) ? lineNumber[j] : ' ';
                screenBuffer[Context.ViewPort.Y + viewY, Context.ViewPort.X + j] = 
                    new ColoredChar(charToRender, color, ConsoleColor.DarkBlue);
            }
        }
    }
    
    /// <summary>
    /// 為指定行繪製文本內容
    /// </summary>
    private void RenderTextForRow(ColoredChar[,] screenBuffer, int viewY, ConsoleText textLine, int lineNumberWidth)
    {
        // 計算文本在螢幕上的位置
        int textX = Context.ViewPort.X + lineNumberWidth;
        int textY = Context.ViewPort.Y + viewY;
        
        // 繪製文本
        RenderText(screenBuffer, textX, textY, textLine, Context.OffsetX, Context.ViewPort, lineNumberWidth);
    }

    public ColoredChar[,] CreateScreenBuffer()
    {
        int height = Console.WindowHeight;
        int width = Console.WindowWidth;
        var screenBuffer = new ColoredChar[height, width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                screenBuffer[y, x] = new ColoredChar(' ', ConsoleColor.White, ConsoleColor.Black);
            }
        }
        return screenBuffer;
    }

    public void WriteToConsole(ColoredChar[,] screenBuffer)
    {
        // 創建一個緩衝區用於收集所有輸出
        var outputBuffer = new StringBuilder();
        outputBuffer.Append($"\x1b[0;0H");
        // 隱藏游標 (符合 Rule 12)
        outputBuffer.Append("\x1b[?25l");

        RenderBufferToConsole(screenBuffer, outputBuffer);
        
        Mode.AfterRender(outputBuffer);

        // 一次性輸出所有內容到控制台
        Console.Write(outputBuffer.ToString());
    }

    /// <summary>
    /// 繪製狀態欄到 screenBuffer
    /// </summary>
    private void RenderStatusBar(ColoredChar[,] screenBuffer)
    {
        int bufferWidth = screenBuffer.GetLength(1);
        int bufferHeight = screenBuffer.GetLength(0);
        
        // 檢查視窗高度是否足夠顯示狀態欄
        if (Context.ViewPort.Y + Context.ViewPort.Height >= bufferHeight)
        {
            return;
        }
        
        // 計算狀態欄的位置
        int statusBarY = Context.ViewPort.Y + Context.ViewPort.Height - 1;
        
        // 檢查 Y 座標是否在 screenBuffer 範圍內
        if (statusBarY < 0 || statusBarY >= bufferHeight)
        {
            return; // Y 座標超出範圍，不繪製
        }

        // 準備狀態欄文本
        string modeName = Mode.GetType().Name.Replace("Vim", "").Replace("Mode", "");
        string statusText = $" {modeName} | Line: {GetActualTextY() + 1} | Col: {GetActualTextX() + 1} ";

        // 更新 StatusBar
        Context.StatusBar = new ConsoleText();
        Context.StatusBar.SetText(0, statusText);
        
        // 設置反色顯示
        for (int i = 0; i < Context.StatusBar.Chars.Length; i++)
        {
            var c = Context.StatusBar.Chars[i];
            if (c == ColoredChar.None) 
            {
                continue; // 跳過中文字符的第二個位置標記
            }
            
            // 如果是空字符，使用空格替代，確保顯示寬度正確
            char displayChar = (c.Char == '\0') ? ' ' : c.Char;
            Context.StatusBar.Chars[i] = new ColoredChar(displayChar, ConsoleColor.Black, ConsoleColor.White);
        }

        // 繪製狀態欄到 screenBuffer
        for (int i = 0; i < Context.StatusBar.Width; i++)
        {
            if (Context.ViewPort.X + i >= 0 && Context.ViewPort.X + i < bufferWidth && i < Context.StatusBar.Chars.Length)
            {
                screenBuffer[statusBarY, Context.ViewPort.X + i] = Context.StatusBar.Chars[i];
            }
        }
    }
    
    /// <summary>
    /// 將 screenBuffer 轉換為 ANSI 控制碼並輸出到 outputBuffer
    /// </summary>
    public void RenderBufferToConsole(ColoredChar[,] screenBuffer, StringBuilder outputBuffer)
    {
        // 輸出整個 screenBuffer 的內容
        for (int y = 0; y < screenBuffer.GetLength(0); y++)
        {
            // 直接輸出每一行的內容
            for (int x = 0; x < screenBuffer.GetLength(1); x++)
            {
                outputBuffer.Append(screenBuffer[y, x].ToAnsiString());
            }
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
                if (textPos < text.Chars.Length)
                {
                    // 直接使用文本中的字符，不做修改
                    var c = text.Chars[textPos];
                    if (c.Char != '\n')
                    {
                        screenBuffer[y, x + i] = c;
                    }
                    else
                    {
                        screenBuffer[y, x + i] = ColoredChar.Empty; // 使用空字符填充
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
                if (pos < screenBuffer.GetLength(1))
                {
                    screenBuffer[y, pos] = ColoredChar.Empty;
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
                screenBuffer.Set(x + i, y, ColoredChar.Empty);
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
        int endX = Math.Min(screenBuffer.GetLength(1) - 1, frameX + frameWidth - 1);
        int endY = Math.Min(screenBuffer.GetLength(0) - 1, frameY + frameHeight - 1);

        // 繪製頂部邊框
        if (startY == frameY) // 頂部邊框在可見範圍內
        {
            // 繪製頂部水平線
            for (int x = startX; x <= endX; x++)
            {
                char c = horizontal;
                if (x == frameX) c = topLeft; // 左上角
                else if (x == frameX + frameWidth - 1) c = topRight; // 右上角
                
                screenBuffer[startY, x] = new ColoredChar(c, frameColor, backgroundColor);
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
                
                screenBuffer[endY, x] = new ColoredChar(c, frameColor, backgroundColor);
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
                    screenBuffer[y, startX] = new ColoredChar(vertical, frameColor, backgroundColor);
                }
            }

            // 右邊框
            if (endX == frameX + frameWidth - 1)
            {
                // 跳過已經繪製的角落
                if (y != frameY && y != frameY + frameHeight - 1)
                {
                    screenBuffer[y, endX] = new ColoredChar(vertical, frameColor, backgroundColor);
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
    
    public void MoveCursorLeft()
    {
        var textX = GetActualTextX();
        if (textX <= 0)
        {
            return;
        }

        var currentLine = GetCurrentLine();
        ColoredChar ch;
        do
        {
            textX--;
            Context.CursorX--;
            if (Context.CursorX < 0)
            {
                Context.CursorX = 0;
                Scroll(-1, 0);
            }
            ch = currentLine.Chars[textX];
        }while(ch.Char == '\0' && textX > 0);
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
        
        // 計算可見區域的有效高度（考慮狀態欄）
        int effectiveViewPortHeight = Context.IsStatusBarVisible ? 
            Context.ViewPort.Height - 1 : Context.ViewPort.Height;
        
        if (scrollDirection > 0)
        {
            // 向下滾動
            needScroll = targetCursorY > Context.ViewPort.Y + effectiveViewPortHeight - 1;
            
            // 如果狀態欄可見，確保游標不會超出有效視口高度
            if (Context.IsStatusBarVisible && 
                targetCursorY >= Context.ViewPort.Y + effectiveViewPortHeight &&
                !needScroll)
            {
                // 將目標游標位置限制在有效視口範圍內
                targetCursorY = Context.ViewPort.Y + effectiveViewPortHeight - 1;
            }
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
        return Context.GetCurrentTextX();
    }

    public int GetActualTextY()
    {
        return Context.GetCurrentTextY();
    }

    public void SetActualTextX(int actualTextX)
    {
        var viewWidth = Context.ViewPort.X + Context.ViewPort.Width - Context.GetLineNumberWidth();
        var cursorX = Context.ViewPort.X + actualTextX + Context.GetLineNumberWidth() - Context.OffsetX;
        Context.CursorX = cursorX;
        if (cursorX > viewWidth)
        {
            if (actualTextX - viewWidth > 0)
            {
                Context.CursorX = Context.ViewPort.X + actualTextX;
            }
            Context.OffsetX = Math.Max(0, actualTextX - viewWidth);
        }
    }

    public ConsoleText GetCurrentLine()
    {
        int index = GetActualTextY();
        
        // 確保索引在有效範圍內
        if (index < 0 || index >= Context.Texts.Count)
        {
            // 如果索引超出範圍，創建一個空行並返回
            if (Context.Texts.Count == 0)
            {
                var newLine = new ConsoleText();
                Context.Texts.Add(newLine);
                return newLine;
            }
            
            // 使用最接近的有效索引
            index = Math.Clamp(index, 0, Context.Texts.Count - 1);
        }
        
        return Context.Texts[index];
    }
}