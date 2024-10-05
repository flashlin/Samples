namespace T1.EfCore.Parsers;

public struct MatchSpan
{
    public bool Success { get; set; }
    public int Index { get; set; }
    public string Value { get; set; }
}