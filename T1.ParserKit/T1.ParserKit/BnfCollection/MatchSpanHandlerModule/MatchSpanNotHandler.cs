namespace T1.ParserKit.BnfCollection.MatchSpanHandlerModule;

public class MatchSpanNotHandler : IMatchSpanHandler
{
    private readonly IMatchSpanHandler _handler;

    public MatchSpanNotHandler(IMatchSpanHandler handler)
    {
        _handler = handler;
    }

    public MatchSpan Match(ReadOnlySpan<char> input, int index)
    {
        var match = _handler.Match(input, index);
        if (match.Success)
        {
            return new MatchSpan
            {
                Success = false,
                Index = index,
                Value = match.Value
            };
        }
        return new MatchSpan
        {
            Success = true,
            Index = index,
            Value = match.Value
        };
    }
}