using T1.EfCore.Parsers;

namespace T1.EFCoreTests;

public class BnfTest
{
    [Test]
    public void Test()
    {
        string bnfGrammar = @"
<expr> ::= <term> ""+"" <expr> | <term>
<term> ::= <factor> ""*"" <term> | <factor>
<factor> ::= ""("" <expr> "")"" | <number>
<number> ::= ""0"" | ""1"" | ""2"" | ""3"" | ""4"" | ""5"" | ""6"" | ""7"" | ""8"" | ""9""
";

        var parser = new BnfParser(bnfGrammar);
        var tree = parser.Parse();

        var text = parser.GetExpressionTreeString(tree);
    }
}