namespace T1.ParserKit.BnfCollection.MatchSpanHandlerModule;

public class MatchSpanPlusHandler : IMatchSpanHandler 
{
    private readonly IMatchSpanHandler _matcher;

    public MatchSpanPlusHandler(IMatchSpanHandler matcher)
    {
        _matcher = matcher;
    }
    
    public MatchSpan Match(ReadOnlySpan<char> input, int index)
    {
        var start = index;
        var match = _matcher.Match(input, index);
        if (!match.Success)
        {
            return match;
        }
        do
        {
            index += match.Value.Length;
            if( index >= input.Length)
            {
                break;
            }
            match = _matcher.Match(input, index);   
        }while (match.Success);
        return new MatchSpan
        {
            Success = true,
            Index = start, 
            Value = input.Slice(start, index - start).ToString()
        };
    }
}