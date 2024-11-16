namespace SqlSharpLit.Common.ParserLit;

public class ParseError : Exception
{
    public static ParseError Empty = new ParseError(string.Empty);
    public ParseError(string message) : base(message)
    {
    }
    public bool IsStart { get; set; }
    public int Offset { get; set; }
}