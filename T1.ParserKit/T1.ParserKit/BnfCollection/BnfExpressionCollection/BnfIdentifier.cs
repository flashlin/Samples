namespace T1.ParserKit.BnfCollection.BnfExpressionCollection;

public class BnfIdentifier : IBnfExpression
{
    public string Name { get; set; } = string.Empty;
}

public class BnfString : IBnfExpression
{
    public string Text { get; set; } = string.Empty;
}

public class BnfRuleIdentifier : IBnfExpression
{
    public string Name { get; set; } = string.Empty;
}
