namespace T1.SqlSharp.Expressions;

public class SqlAssignExpr : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.AssignExpr;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AssignExpr(this);
    }

    public required ISqlExpression Left { get; set; }
    public required ISqlExpression Right { get; set; }

    public string ToSql()
    {
        return $"{Left.ToSql()} = {Right.ToSql()}";
    }
}