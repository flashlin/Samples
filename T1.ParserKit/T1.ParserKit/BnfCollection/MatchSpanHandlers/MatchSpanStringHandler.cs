namespace T1.ParserKit.BnfCollection.MatchSpanHandlerModule;

public class MatchSpanStringHandler : IMatchSpanHandler
{
    private readonly string _pattern;
    private readonly MatchSpanCharHandler[] _handlers;

    public MatchSpanStringHandler(string pattern)
    {
        _pattern = pattern;
        _handlers = new MatchSpanCharHandler[pattern.Length];
        foreach (var item in pattern.Select((x,index) => (Value: x,Index: index)))
        {
            _handlers[item.Index] = new MatchSpanCharHandler(item.Value);
        }
    }
    
    public MatchSpan Match(ReadOnlySpan<char> input, int index)
    {
        var start = index;
        var count = 0;
        foreach (var handler in _handlers)
        {
            var match = handler.Match(input, index);
            count++;
            if (!match.Success)
            {
                return new MatchSpan
                {
                    Success = false,
                    Index = start,
                    Value = input.Slice(start, count).ToString()
                };
            }
            index++;
        }
        return new MatchSpan
        {
            Success = true,
            Index = start, 
            Value = input.Slice(start, count).ToString()
        };
    }
}