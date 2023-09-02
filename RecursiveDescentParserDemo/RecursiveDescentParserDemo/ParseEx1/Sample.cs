namespace RecursiveDescentParserDemo.ParseEx1;

public class Sample
{
    public void Run()
    {
        // 示例用法
        var input = "cababab";
        //Test1(input);
        Method2(input);
    }

    public void Method1(string input)
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

    public void Method2(string input)
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