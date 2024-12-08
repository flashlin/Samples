namespace T1.SqlSharp.Expressions;

public class SqlAssignExpr : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.AssignExpr;
    public required ISqlExpression Left { get; set; }
    public required ISqlExpression Right { get; set; }

    public string ToSql()
    {
        return $"{Left.ToSql()} = {Right.ToSql()}";
    }
}