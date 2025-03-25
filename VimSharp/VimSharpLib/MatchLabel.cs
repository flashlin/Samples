namespace VimSharpLib;

public class MatchLabel
{
    public int X { get; set; }
    public int Y { get; set; }
    public string Label { get; set; }

    public MatchLabel(int x, int y, string label)
    {
        X = x;
        Y = y;
        Label = label;
    }
}