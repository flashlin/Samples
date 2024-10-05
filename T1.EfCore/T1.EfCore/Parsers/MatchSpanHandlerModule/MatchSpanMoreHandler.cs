namespace T1.EfCore.Parsers.MatchSpanHandlerModule;

public class MatchSpanMoreHandler : IMatchSpanHandler 
{
    private readonly IMatchSpanHandler _matcher;

    public MatchSpanMoreHandler(IMatchSpanHandler matcher)
    {
        _matcher = matcher;
    }
    
    public MatchSpan Match(ReadOnlySpan<char> input, int index)
    {
        var start = index;
        var match = _matcher.Match(input, index);
        if (!match.Success)
        {
            return new MatchSpan
            {
                Success = true,
                Index = index,
                Value = match.Value
            };
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