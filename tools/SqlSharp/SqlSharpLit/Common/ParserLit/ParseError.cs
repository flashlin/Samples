namespace SqlSharpLit.Common.ParserLit;

public class ParseError : Exception
{
    public static readonly ParseError Empty = new(string.Empty);
    public ParseError(string message) : base(message)
    {
    }
    public bool IsStart { get; set; }
    public int Offset { get; set; }
}