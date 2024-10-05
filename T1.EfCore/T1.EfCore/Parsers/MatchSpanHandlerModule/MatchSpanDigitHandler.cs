namespace T1.EfCore.Parsers.MatchSpanHandlerModule;

public class MatchSpanDigitHandler : IMatchSpanHandler 
{
    public MatchSpan Match(ReadOnlySpan<char> input, int index)
    {
        if (!char.IsDigit(input[index]))
        {
            return new MatchSpan
            {
                Success = false,
                Index = index,
                Value = input.Slice(index, 1).ToString()
            };
        } 
        var start = index;
        index++;
        return new MatchSpan
        {
            Success = true,
            Index = start, 
            Value = input.Slice(start, index - start).ToString()
        };
    }
}