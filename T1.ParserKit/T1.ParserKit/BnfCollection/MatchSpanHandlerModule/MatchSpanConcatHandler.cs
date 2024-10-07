namespace T1.ParserKit.BnfCollection.MatchSpanHandlerModule;

public class MatchSpanConcatHandler : IMatchSpanHandler
{
    private readonly IMatchSpanHandler _handler;
    private readonly IMatchSpanHandler _nextHandler;

    public MatchSpanConcatHandler(IMatchSpanHandler handler, IMatchSpanHandler nextHandler)
    {
        _handler = handler;
        _nextHandler = nextHandler;
    }

    public MatchSpan Match(ReadOnlySpan<char> input, int index)
    {
        var match = _handler.Match(input, index);
        if (!match.Success)
        {
            return match;
        }
        var nextMatch = _nextHandler.Match(input, match.Index + match.Value.Length);
        if (!nextMatch.Success)
        {
            return new MatchSpan
            {
                Success = false,
                Index = match.Index,
                Value = input.Slice(match.Index, match.Value.Length + nextMatch.Value.Length).ToString()
            };
        }
        return new MatchSpan
        {
            Success = true,
            Index = match.Index,
            Value = input.Slice(match.Index, match.Value.Length + nextMatch.Value.Length).ToString()
        };
    }
}