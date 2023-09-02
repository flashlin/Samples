namespace RecursiveDescentParserDemo.ParseEx1;

public class MiniCfg : CfgBase<string>
{
    MatchToken A = new MatchToken("a");
    MatchToken B = new MatchToken("b");
    MatchToken C = new MatchToken("c");

    public List<Token<string>> Start()
    {
        var tokens = Or(
            c => c.Consume(A).Consume(B),
            c => c.Consume(B).Consume(A),
            c => SubRule(E)
        );
        var subTokens = Or(
            c => c.Consume(A).Consume(B),
            c => c.Consume(B).Consume(A)
        );
        tokens.AddRange(subTokens);
        return tokens;
    }

    ContextFreeGrammer<string> E()
    {
        return Consume(C);
    }

    private ContextFreeGrammer<string> SubRule(Func<ContextFreeGrammer<string>> func)
    {
        return func();
    }
}