namespace T1.ParserKit.BnfCollection.BnfExpressionCollection;

public class BnfGroup : IBnfExpression
{
    public required IBnfExpression InnerExpression { get; set; }
}