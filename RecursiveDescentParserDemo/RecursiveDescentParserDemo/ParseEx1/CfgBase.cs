namespace RecursiveDescentParserDemo.ParseEx1;

public class CfgBase<T>
{
    readonly ContextFreeGrammer<T> _cfg = new();

    public void SetInput(IEnumerableStream<T> input)
    {
        _cfg.SetInput(input);
    }

    protected List<Token<T>> Or(params Func<ContextFreeGrammer<T>, ContextFreeGrammer<T>>[] rules)
    {
        return _cfg.Or(rules);
    }

    protected ContextFreeGrammer<T> Consume(IMatcher<T> matchToken)
    {
        return _cfg.Consume(matchToken);
    }
}