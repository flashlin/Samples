using T1.EfCore.Parsers.MatchSpanHandlerModule;

namespace T1.EfCore.Parsers;

public class BnfTokenizer
{
    private readonly IMatchSpanHandler[] _matchSpanHandlers;

    public BnfTokenizer()
    {
        _matchSpanHandlers = 
        [
            Char('"').Concat(
                String("\\\"").Or(Not("\"")).More()
                ).Concat(Char('"')),
            String("::="),
            Digit().Plus(),
            String("<").Concat(Not(">").Plus()).Concat(String(">")),
            String("|"),
            String("("),
            String(")"),
            Letter().Or(String("_")).Plus().Concat(Letter().Or(String("_")).Or(Digit()).More()),
        ];
    }

    private MatchSpanNotHandler Not(string pattern)
    {
        return new MatchSpanNotHandler(String(pattern));
    }
    
    private MatchSpanNotHandler Not(IMatchSpanHandler pattern)
    {
        return new MatchSpanNotHandler(pattern);
    }

    public List<MatchSpan> ExtractMatches(string input)
    {
        var index = 0;
        var inputSpan = input.AsSpan();
        var results = new List<MatchSpan>();
        while (index < inputSpan.Length)
        {
            index = SkipWhitespace(inputSpan, index);
            var match = MatchRules(inputSpan, index);
            if (!match.Success)
            {
                throw new Exception($"Unexpected text '{inputSpan.Slice(index).ToString()}' at position {index}");
            }
            results.Add(match);
            index = match.Index + match.Value.Length;
        }
        return results;
    }

    private MatchSpanCharFuncHandler Digit()
    {
        return new MatchSpanCharFuncHandler(char.IsDigit);
    }
    
    private MatchSpanCharFuncHandler Letter()
    {
        return new MatchSpanCharFuncHandler(char.IsLetter);
    }

    private MatchSpan MatchRules(ReadOnlySpan<char> input, int index)
    {
        foreach (var matchSpanHandler in _matchSpanHandlers)
        {
            var match = matchSpanHandler.Match(input, index);
            if (match.Success)
            {
                return match;
            }
        }
        return new MatchSpan
        {
            Success = false,
            Index = index,
            Value = string.Empty
        };
    }

    private int SkipWhitespace(ReadOnlySpan<char> input, int index)
    {
        while (index < input.Length && char.IsWhiteSpace(input[index]))
        {
            index++;
        }
        return index;
    }

    private MatchSpanStringHandler String(string pattern)
    {
        return new MatchSpanStringHandler(pattern);
    }
    
    private MatchSpanCharHandler Char(char pattern)
    {
        return new MatchSpanCharHandler(pattern);
    }
}