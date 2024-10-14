namespace T1.ParserKit.BnfCollection.BnfExpressionCollection;

public class BnfBinaryExpression : IBnfExpression
{
    public required IBnfExpression Left { get; set; }
    public string Operator { get; set; } = string.Empty; 
    public required IBnfExpression Right { get; set; }
}