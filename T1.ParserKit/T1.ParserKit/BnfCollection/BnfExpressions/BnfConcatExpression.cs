namespace T1.ParserKit.BnfCollection.BnfExpressions;

public class BnfConcatExpression : IBnfExpression
{
    public List<IBnfExpression> Items { get; set; } = [];
}