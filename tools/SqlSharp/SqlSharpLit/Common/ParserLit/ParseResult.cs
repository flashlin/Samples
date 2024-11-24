namespace SqlSharpLit.Common.ParserLit;

public class ParseResult<T> : IParseResult
{
    public static implicit operator ParseResult<T>(ParseError error)
    {
        return new ParseResult<T>(error);
    }
    
    public static implicit operator ParseResult<T>(T? result)
    {
        return new ParseResult<T>(result);
    }
    
    public ParseResult<T1> To<T1>()
        where T1 : class
    {
        if (HasError)
        {
            return Error;
        }
        return new ParseResult<T1>((T1?)Object);
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

    public bool HasValue
    {
        get
        {
            if(HasResult)
            {
                return Result != null;
            }

            return false;
        }
    }

    public ParseError Error { get; set; } = ParseError.Empty;
    public bool HasError { get; set; }
}