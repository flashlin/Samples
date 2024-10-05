namespace T1.EfCore.Parsers;

public class BnfTokenizer
{
    delegate MatchSpan MatchSpanFunc(ReadOnlySpan<char> input, int index);
    private readonly MatchSpanFunc[] _matchSpanFuncs;

    public BnfTokenizer()
    {
        _matchSpanFuncs =
        [
            MatchDigits
        ]; 
    }

    public List<MatchSpan> ExtractMatches(string input)
    {
        var index = 0;
        var inputSpan = input.AsSpan();
        var results = new List<MatchSpan>();
        while (index < inputSpan.Length)
        {
            index = SkipWhitespace(inputSpan, index);
            var match = MatchSpan.Empty;
            foreach (var matchSpanFunc in _matchSpanFuncs)
            {
                match = matchSpanFunc(inputSpan, index);
                if (match.Success)
                {
                    break;
                }
            }
            if (!match.Success)
            {
                throw new Exception($"Unexpected text '{inputSpan.Slice(index).ToString()}' at position {index}");
            }
            results.Add(match);
            index = match.Index + match.Value.Length;
        }
        return results;
    }
    
    private int SkipWhitespace(ReadOnlySpan<char> input, int index)
    {
        while (index < input.Length && char.IsWhiteSpace(input[index]))
        {
            index++;
        }
        return index;
    }

    private MatchSpan MatchDigits(ReadOnlySpan<char> input, int index)
    {
        if (!char.IsDigit(input[index]))
        {
            return MatchSpan.Empty;
        }
        var start = index;
        while (index < input.Length && char.IsDigit(input[index]))
        {
            index++;
        }
        return new MatchSpan
        {
            Index = start, 
            Value = input.Slice(start, index - start).ToString()
        };
    }
}