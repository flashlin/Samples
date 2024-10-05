namespace T1.EfCore.Parsers.MatchSpanHandlerModule;

public class MatchSpanCharHandler : IMatchSpanHandler 
{
    private readonly char _pattern;

    public MatchSpanCharHandler(char pattern)
    {
        _pattern = pattern;
    }
    
    public MatchSpan Match(ReadOnlySpan<char> input, int index)
    {
        if (input[index] != _pattern)
        {
            return MatchSpan.Empty;
        } 
        var start = index;
        index++;
        return new MatchSpan
        {
            Index = start, 
            Value = input.Slice(start, index - start).ToString()
        };
    }
}