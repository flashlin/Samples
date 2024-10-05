namespace T1.EfCore.Parsers.MatchSpanHandlerModule;

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
        foreach (var handler in _handlers)
        {
            var match = handler.Match(input, index);
            if (!match.Success)
            {
                return MatchSpan.Empty;
            }
            index++;
        }
        return new MatchSpan
        {
            Index = start, 
            Value = input.Slice(start, index - start).ToString()
        };
    }
}