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
    public string StatusBarText { get; set; } = "";
    public ConsoleText StatusBar { get; set; } = new();

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
}