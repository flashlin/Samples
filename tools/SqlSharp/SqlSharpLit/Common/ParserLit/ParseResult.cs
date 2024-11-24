namespace SqlSharpLit.Common.ParserLit;

public class ParseResult<T> : IParseResult
{
    public static ParseResult<T> From<T1>(ParseResult<T1> result)
    {
        if (result.HasResult)
        {
            return new ParseResult<T>((T?)result.Object);
        }
        return new ParseResult<T>(result.Error);
    }
    
    public ParseResult(T? result)
    {
        HasResult = true;
        Result = result;
    }

    public ParseResult(ParseError error)
    {
        HasError = true;
        Error = error;
    }

    public T? Result { get; set; }

    public T ResultValue
    {
        get
        {
            if (Result == null)
            {
                throw new InvalidOperationException("Result is null");
            }
            return Result;
        }
    }
    
    public object? Object => Result;

    public object ObjectValue
    {
        get
        {
            if (Result == null)
            {
                throw new InvalidOperationException("Result is null");
            }
            return Result;
        }
    }

    public bool HasResult { get; set; }
    public ParseError Error { get; set; } = ParseError.Empty;
    public bool HasError { get; set; }
}