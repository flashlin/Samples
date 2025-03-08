namespace VimSharpLib;

public class ConsoleContext
{
    public int CursorX { get; set; }
    public int CursorY { get; set; }
    public List<ConsoleText> Texts { get; set; } = [];

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