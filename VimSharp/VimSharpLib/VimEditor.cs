namespace VimSharpLib;
using System.Text;

public class VimEditor
{

    public VimEditor()
    {
        Initialize();
    }

    public bool IsRunning { get; set; } = true;
    public ConsoleContext Context { get; set; } = new();
    public IVimMode Mode { get; set; }

    public void Initialize()
    {
        Mode = new VimVisualMode { Instance = this };
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
        
        // 只繪製可見區域內的行
        for (var i = 0; i < visibleLines; i++)
        {
            // 計算實際要繪製的文本行索引
            int textIndex = Context.OffsetY + i;
            
            // 確保索引有效
            if (textIndex >= 0 && textIndex < Context.Texts.Count)
            {
                var text = Context.Texts[textIndex];
                
                // 確保文本寬度足夠
                if (text.Width < Context.ViewPort.Width + Context.OffsetX)
                {
                    text.Width = Context.ViewPort.Width + Context.OffsetX;
                }
                
                // 直接繪製文本，考慮 ViewPort 和偏移量
                RenderText(Context.ViewPort.X, Context.ViewPort.Y + i, text, Context.OffsetX, Context.ViewPort);
            }
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
            Console.SetCursorPosition(cursorScreenX, cursorScreenY);
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
        Console.SetCursorPosition(x, y);
        
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
                if (c.Char != '\0')
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
            // 創建一個黑底白字的空格
            var emptyChar = new ColoredChar(' ', ConsoleColor.White, ConsoleColor.Black);
            
            // 填充空白字符
            for (int i = 0; i < paddingCount; i++)
            {
                sb.Append(emptyChar.ToAnsiString());
            }
        }
        
        // 輸出構建好的字符串
        Console.Write(sb.ToString());
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
}