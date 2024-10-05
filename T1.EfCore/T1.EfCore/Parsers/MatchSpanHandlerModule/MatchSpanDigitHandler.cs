namespace T1.EfCore.Parsers.MatchSpanHandlerModule;

public class MatchSpanDigitHandler : IMatchSpanHandler 
{
    public MatchSpan Match(ReadOnlySpan<char> input, int index)
    {
        if (!char.IsDigit(input[index]))
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