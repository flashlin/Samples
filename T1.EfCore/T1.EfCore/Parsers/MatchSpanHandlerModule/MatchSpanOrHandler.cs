namespace T1.EfCore.Parsers.MatchSpanHandlerModule;

public class MatchSpanOrHandler : IMatchSpanHandler
{
    private readonly IMatchSpanHandler _handler1;
    private readonly IMatchSpanHandler _handler2;

    public MatchSpanOrHandler(IMatchSpanHandler handler1, IMatchSpanHandler handler2)
    {
        _handler1 = handler1;
        _handler2 = handler2;
    }
    
    public MatchSpan Match(ReadOnlySpan<char> input, int index)
    {
        var match1 = _handler1.Match(input, index);
        if (match1.Success)
        {
            return match1;
        }
        var match2 = _handler2.Match(input, index);
        if (match2.Success)
        {
            return match2;
        }
        if(match1.Value.Length > match2.Value.Length)
        {
            return new MatchSpan
            {
                Success = false,
                Index = index,
                Value = match1.Value
            };
        }
        return new MatchSpan
        {
            Success = false,
            Index = index,
            Value = match2.Value
        };
    }
}