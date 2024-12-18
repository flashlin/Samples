namespace T1.SqlSharp.Expressions;

public class SqlArithmeticBinaryExpr : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.ArithmeticBinaryExpr;
    public TextSpan Span { get; set; } = new();

    public required ISqlExpression Left { get; set; }
    public ArithmeticOperator Operator { get; set; }
    public required ISqlExpression Right { get; set; }

    public string ToSql()
    {
        return $"{Left.ToSql()} {Operator.ToSql()} {Right.ToSql()}";
    }
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ArithmeticBinaryExpr(this);
    }
}