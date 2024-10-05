namespace T1.EfCore.Parsers.MatchSpanHandlerModule;

public class MatchSpanCharFuncHandler : IMatchSpanHandler 
{
    private readonly Func<char, bool> _isChar;

    public MatchSpanCharFuncHandler(Func<char, bool> isChar)
    {
        _isChar = isChar;
    }
    
    public MatchSpan Match(ReadOnlySpan<char> input, int index)
    {
        if (!_isChar(input[index]))
        {
            return new MatchSpan
            {
                Success = false,
                Index = index,
                Value = input.Slice(index, 1).ToString()
            };
        } 
        return new MatchSpan
        {
            Success = true,
            Index = index, 
            Value = input.Slice(index, 1).ToString()
        };
    }
}