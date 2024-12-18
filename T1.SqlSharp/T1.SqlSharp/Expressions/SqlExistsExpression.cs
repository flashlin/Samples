namespace T1.SqlSharp.Expressions;

public class SqlExistsExpression : ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.ExistsExpression;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ExistsExpression(this);
    }

    public required ISqlExpression Query { get; set; }
    public string ToSql()
    {
        return $"EXISTS ({Query.ToSql()})";
    }
}