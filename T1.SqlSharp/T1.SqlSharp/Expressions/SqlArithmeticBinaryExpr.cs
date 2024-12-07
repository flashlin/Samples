namespace T1.SqlSharp.Expressions;

public class SqlArithmeticBinaryExpr : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.ArithmeticBinaryExpr;
    public required ISqlExpression Left { get; set; }
    public string Operator { get; set; } = "+";
    public required ISqlExpression Right { get; set; }

    public string ToSql()
    {
        return $"{Left.ToSql()} {Operator} {Right.ToSql()}";
    }
}