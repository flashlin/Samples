namespace VimSharpLib;

public class ConsoleContext
{
    public int CursorX { get; set; }
    public int CursorY { get; set; }
    public List<ConsoleText> Texts { get; set; } = [];
    
    /// <summary>
    /// 控制台視窗的可視矩形區域
    /// </summary>
    public ConsoleRectangle ViewPort { get; set; } = new ConsoleRectangle();

    public bool IsStatusBarVisible { get; set; } = false;
    public string StatusBarText { get; set; } = "";
    public ConsoleText StatusBar { get; set; } = new ConsoleText();

    /// <summary>
    /// 文本內容的水平偏移量
    /// </summary>
    public int OffsetX { get; set; } = 0;
    
    /// <summary>
    /// 文本內容的垂直偏移量
    /// </summary>
    public int OffsetY { get; set; } = 0;

    public void SetText(int x, int y, string text)
    {
        if (Texts.Count <= y)
        {
            Texts.Add(new ConsoleText());
        }
        var consoleText = Texts[y];
        consoleText.SetText(x, text);
    }

    public ConsoleText GetText(int y)
    {
        if (Texts.Count <= y)
        {
            Texts.Add(new ConsoleText());
        }
        return Texts[y];
    }

    /// <summary>
    /// 設置視窗的矩形區域
    /// </summary>
    /// <param name="x">視窗左上角的 X 座標</param>
    /// <param name="y">視窗左上角的 Y 座標</param>
    /// <param name="width">視窗的寬度</param>
    /// <param name="height">視窗的高度</param>
    public void SetViewPort(int x, int y, int width, int height)
    {
        ViewPort = new ConsoleRectangle(x, y, width, height);
    }
}