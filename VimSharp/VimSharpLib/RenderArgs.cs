namespace VimSharpLib;

public class RenderArgs
{
    public int X { get; set; }
    public int Y { get; set; }
    public required ConsoleText Text { get; set; }
}