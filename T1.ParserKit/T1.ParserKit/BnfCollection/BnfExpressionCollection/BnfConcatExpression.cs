namespace T1.ParserKit.BnfCollection.BnfExpressionCollection;

public class BnfConcatExpression : IBnfExpression
{
    public List<IBnfExpression> Items { get; set; } = [];
}