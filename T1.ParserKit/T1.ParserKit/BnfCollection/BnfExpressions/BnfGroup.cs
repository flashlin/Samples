namespace T1.ParserKit.BnfCollection.BnfExpressions;

public class BnfGroup : IBnfExpression
{
    public required IBnfExpression InnerExpression { get; set; }
}