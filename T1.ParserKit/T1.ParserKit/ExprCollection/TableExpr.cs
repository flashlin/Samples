namespace T1.ParserKit.ExprCollection;

public class TableExpr : SqlExpr
{
    public string Name { get; set; } = string.Empty;
    public string AliasName { get; set; } = string.Empty;
}