namespace VimSharpLib;

public class ConsoleContext
{
    public int CursorX { get; set; }
    public int CursorY { get; set; }
    public List<ConsoleText> Texts { get; set; } = [];
    
    /// <summary>
    /// 控制台視窗的可視矩形區域
    /// </summary>
    public ViewArea ViewPort { get; set; } = new();

    public bool IsStatusBarVisible { get; set; } = false;
    public ConsoleText StatusBar { get; set; } = new();

    /// <summary>
    /// 文本內容的水平偏移量
    /// </summary>
    public int OffsetX { get; set; } = 0;
    
    /// <summary>
    /// 文本內容的垂直偏移量
    /// </summary>
    public int OffsetY { get; set; } = 0;
    
    /// <summary>
    /// 是否顯示相對行號
    /// </summary>
    public bool IsLineNumberVisible { get; set; } = false;

    public int StatusBarHeight => IsStatusBarVisible ? 1 : 0;

    /// <summary>
    /// 設置視窗的矩形區域並調整游標位置
    /// </summary>
    /// <param name="x">視窗左上角的 X 座標</param>
    /// <param name="y">視窗左上角的 Y 座標</param>
    /// <param name="width">視窗的寬度</param>
    /// <param name="height">視窗的高度</param>
    public void SetViewPort(int x, int y, int width, int height)
    {
        ViewPort = new ViewArea(x, y, width, height);
        CursorX = x + GetLineNumberWidth();
        CursorY = y;
    }

    /// <summary>
    /// 計算相對行號區域的寬度
    /// </summary>
    /// <returns>相對行號區域的寬度</returns>
    public int GetLineNumberWidth()
    {
        if (!IsLineNumberVisible)
        {
            return 0;
        }
        return Texts.Count.ToString().Length + 1;
    }

    public void SetText(int x, int y, string text)
    {
        if (Texts.Count <= y)
        {
            Texts.Add(new ConsoleText());
        }
        var consoleText = Texts[y];
        consoleText.SetText(x, text);
    }

    public ConsoleText GetText(int textY)
    {
        // 確保 textY 在有效範圍內
        while (textY >= Texts.Count)
        {
            Texts.Add(new ConsoleText());
        }
        return Texts[textY];
    }

    public int GetCursorLeft()
    {
        return ViewPort.X + GetLineNumberWidth();
    }

    public int GetCurrentTextX()
    {
        return ComputeTextX(CursorX);
    }

    public int ComputeTextX(int cursorX)
    {
        return cursorX - ViewPort.X + OffsetX - GetLineNumberWidth();
    }

    public int GetCurrentTextY()
    {
        return ComputeTextY(CursorY);
    }

    public int ComputeTextY(int cursorY)
    {
        return cursorY - ViewPort.Y + OffsetY;
    }

    public int ComputeOffset(int viewTextX, int viewTextY)
    {
        var offset = 0;
        for (var i = 0; i < viewTextY; i++)
        {
            offset += Texts[i].Width;
        }
        offset += viewTextX;
        return offset;
    }
}