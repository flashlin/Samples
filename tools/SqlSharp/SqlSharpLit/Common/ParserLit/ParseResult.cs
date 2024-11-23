namespace SqlSharpLit.Common.ParserLit;

public class ParseResult<T>
{
    public ParseResult(T result)
    {
        HasResult = true;
        Result = result;
    }

    public ParseResult(ParseError error)
    {
        HasError = true;
        Error = error;
    }

    public T Result { get; set; }
    public bool HasResult { get; set; }
    public ParseError Error { get; set; } = ParseError.Empty;
    public bool HasError { get; set; }
}