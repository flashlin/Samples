using FluentAssertions;
using T1.EfCore.Parsers;

namespace T1.EFCoreTests;

public class BnfTokenizerTest
{
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

    [Test]
    public void Or()
    {
        var matches = WhenExtractMatches("|");
        matches.Should().BeEquivalentTo([
            new MatchSpan
            {
                Success = true,
                Index = 0,
                Value = "|"
            }
        ]);
    }

    [Test]
    public void QuotesString()
    {
        var matches = WhenExtractMatches("\"ABC\\\"123\"");
        matches.Should().BeEquivalentTo([
            new MatchSpan
            {
                Success = true,
                Index = 0,
                Value = "\"ABC\\\"123\""
            }
        ]);
    }


    [Test]
    public void LBracket()
    {
        var matches = WhenExtractMatches("(");
        matches.Should().BeEquivalentTo([
            new MatchSpan
            {
                Success = true,
                Index = 0,
                Value = "("
            }
        ]);
    }


    [Test]
    public void RBracket()
    {
        var matches = WhenExtractMatches(")");
        matches.Should().BeEquivalentTo([
            new MatchSpan
            {
                Success = true,
                Index = 0,
                Value = ")"
            }
        ]);
    }

    [Test]
    public void Rule()
    {
        var matches = WhenExtractMatches("<rule1> ::= \"1\"");
        matches.Should().BeEquivalentTo([
            new MatchSpan
            {
                Success = true,
                Index = 0,
                Value = "<rule1>"
            },
            new MatchSpan
            {
                Success = true,
                Index = 8,
                Value = "::="
            },
            new MatchSpan
            {
                Success = true,
                Index = 12,
                Value = "\"1\""
            },
        ]);
    }

    private static MatchSpan[] WhenExtractMatches(string input)
    {
        var tokenizer = new BnfTokenizer();
        var matches = tokenizer.ExtractMatches(input).ToArray();
        return matches;
    }
}