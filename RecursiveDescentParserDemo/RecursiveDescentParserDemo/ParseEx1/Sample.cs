namespace RecursiveDescentParserDemo.ParseEx1;

public class ParserBase<T>
{
    ContextFreeGrammer<T> _cfg = new();

    public void SetInput(IEnumerableStream<T> input)
    {
        _cfg.SetInput(input);
    }

    public List<Token<T>> Or(params Func<ContextFreeGrammer<T>, ContextFreeGrammer<T>>[] rules)
    {
        return _cfg.Or(rules);
    }

    protected ContextFreeGrammer<T> Consume(IMatcher<T> matchToken)
    {
        return _cfg.Consume(matchToken);
    }
}

public class MiniCfg : ParserBase<string>
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

public class Sample
{
    public void Run()
    {
        // 示例用法
        var input = "cababab";
        //Test1(input);
        Test2(input);
    }

    public void Test1(string input)
    {
        var A = new MatchToken("a");
        var B = new MatchToken("b");

        var cfg = new ContextFreeGrammer<string>(new TextEnumerableStream(input));
        Func<List<Token<string>>> start = () =>
        {
            return cfg.Or(
                c => c.Consume(A).Consume(B),
                c => c.Consume(B).Consume(A)
            );
        };
        var tokens = start();

        Console.WriteLine(tokens.Count);
        foreach (var token in tokens)
        {
            Console.WriteLine(token.Value);
        }
    }

    public void Test2(string input)
    {
        var m = new MiniCfg();
        m.SetInput(new TextEnumerableStream(input));
        var tokens = m.Start();
        Console.WriteLine(tokens.Count);
        foreach (var token in tokens)
        {
            Console.WriteLine(token.Value);
        }
    }
}