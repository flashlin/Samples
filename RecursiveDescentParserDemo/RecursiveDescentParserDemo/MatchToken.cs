using RecursiveDescentParserDemo.ParseEx1;

namespace RecursiveDescentParserDemo;

public class MatchToken : IMatcher<string>
{
    private string _keyword;

    public MatchToken(string keyword)
    {
        _keyword = keyword;
    }

    public bool Match(string input)
    {
        return input == _keyword;
    }
}