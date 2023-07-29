namespace T1.ParserKit.ExprCollection;

public class FromSourceExpr : SqlExpr
{
    public SqlExpr Clause { get; set; } = null!;
    public string AliasName { get; set; } = string.Empty;
}