namespace T1.ParserKit.BnfCollection.BnfExpressionCollection;

public class BnfRule : IBnfExpression
{
    public string RuleName { get; set; } = string.Empty;
    public List<IBnfExpression> Expressions { get; set; } = [];
}