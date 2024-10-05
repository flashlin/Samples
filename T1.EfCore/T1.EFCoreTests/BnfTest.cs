using FluentAssertions;
using T1.EfCore.Parsers;

namespace T1.EFCoreTests;

public class BnfTest
{
    [Test]
    public void Test()
    {
        string bnfGrammar = @"
<expr> ::= <term> + <expr> | <term>
<term> ::= <factor> * <term> | <factor>
<factor> ::= ( <expr> ) | <number>
<number> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
";

        var parser = new BnfParser(bnfGrammar);
        var tree = parser.Parse();

        var text = parser.GetExpressionTreeString(tree);
    }
}

public class BnfTokenizerTest
{
    [Test]
    public void Digits()
    {
        var tokenizer = new BnfTokenizer();
        var input = "123";
        var matches = tokenizer.ExtractMatches(input).ToArray();
        matches.Should().BeEquivalentTo([
            new MatchSpan { Index = 0, Value = "123" }
        ]);
    }
}