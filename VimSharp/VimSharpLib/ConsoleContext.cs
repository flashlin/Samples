namespace VimSharpLib;

public class ConsoleContext
{
    public int X { get; set; }
    public int Y { get; set; }
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
}