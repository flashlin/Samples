namespace T1.EfCore.Parsers.MatchSpanHandlerModule;

public class MatchSpanCharHandler : IMatchSpanHandler
{
    private readonly char _ch;

    public MatchSpanCharHandler(char ch)
    {
        _ch = ch;
    }

    public MatchSpan Match(ReadOnlySpan<char> input, int index)
    {
        if (input[index] != _ch)
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