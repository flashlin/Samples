namespace T1.SqlSharp.Expressions;

public class SqlAliasExpr : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.AliasExpr;
    public TextSpan Span { get; set; } = new();
    public required string Name { get; set; } = string.Empty;
    public string ToSql()
    {
        return $"AS {Name}";
    }
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AliasExpr(this);
    }
}