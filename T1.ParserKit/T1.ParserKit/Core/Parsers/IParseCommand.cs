namespace T1.ParserKit.Core.Parsers
{
    public interface IParseCommand<TInput, TValue>
    {
        ParseResult<TInput, TValue>? Execute(TInput input);
    }
}