using System.Text;

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

    public string GetText(int offset, int length)
    {
        var text = new StringBuilder();
        var isStart = true;
        var currentLineOffset = 0;
        var cutLength = length;
        for (var i = 0; i < Texts.Count; i++)
        {
            if (cutLength <= 0)
            {
                break;
            }
            var currentLine = Texts[i];
            if (isStart && currentLineOffset + currentLine.Width >= offset)
            {
                var line = currentLine.GetChars(offset - currentLineOffset, cutLength);
                text.Append(line.ToText());
                cutLength -= line.Length;
                isStart = false;
                continue;
            }
            var line2 = currentLine.GetChars(0, cutLength);
            text.Append(line2.ToText());
            cutLength -= line2.Length + 1;
        }
        return text.ToString();
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
            offset += Texts[i].Width + 1;
        }
        offset += viewTextX;
        return offset;
    }
}