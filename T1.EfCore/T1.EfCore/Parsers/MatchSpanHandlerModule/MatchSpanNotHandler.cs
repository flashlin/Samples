namespace T1.EfCore.Parsers.MatchSpanHandlerModule;

public class MatchSpanNotHandler : IMatchSpanHandler
{
    private readonly MatchSpanStringHandler _handler;

    public MatchSpanNotHandler(MatchSpanStringHandler handler)
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