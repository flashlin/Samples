namespace RecursiveDescentParserDemo.ParseEx1;

public class Sample
{
    public void Run()
    {
        // 示例用法
        var input = "ababab";

        var A = new MatchToken("a");
        var B = new MatchToken("b");
        
        var cfg = new ContextFreeGrammer<string>(new TextEnumerableStream(input));
        var tokens = cfg.Or(
            c => c.Consume(A).Consume(B),
            c => c.Consume(B).Consume(A)
        );

        Console.WriteLine(tokens.Count);
        foreach (var token in tokens)
        {
            Console.WriteLine(token.Value);
        }
    }
}