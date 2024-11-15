namespace SqlSharpLit.Common.ParserLit;

public class ParseError : Exception
{
    public ParseError(string message) : base(message)
    {
    }
}