using FluentAssertions;
using T1.EfCore.Parsers;

namespace T1.EFCoreTests;

public class BnfTokenizerTest
{
    [Test]
    public void Digits()
    {
        var matches = WhenExtractMatches("123");
        matches.Should().BeEquivalentTo([
            new MatchSpan
            {
                Success = true,
                Index = 0, 
                Value = "123"
            }
        ]);
    }
    
    [Test]
    public void RuleEqual()
    {
        var matches = WhenExtractMatches("::=");
        matches.Should().BeEquivalentTo([
            new MatchSpan
            {
                Success = true,
                Index = 0, 
                Value = "::="
            }
        ]);
    }

    private static MatchSpan[] WhenExtractMatches(string input)
    {
        var tokenizer = new BnfTokenizer();
        var matches = tokenizer.ExtractMatches(input).ToArray();
        return matches;
    }
}