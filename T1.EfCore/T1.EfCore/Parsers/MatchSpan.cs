namespace T1.EfCore.Parsers;

public struct MatchSpan
{
    public bool Success => Index >= 0;
    public int Index { get; set; }
    public string Value { get; set; }
    public static MatchSpan Empty => new MatchSpan { Index = -1, Value = string.Empty };
}