namespace T1.SqlSharp.Expressions;

public class SqlUnaryExpr : ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.UnaryExpression;
    public UnaryOperator Op { get; set; } = UnaryOperator.BitwiseNot;
    public required ISqlExpression Operand { get; set; }

    public string ToSql()
    {
        return $"{Op.ToSql()} {Operand.ToSql()}";
    }
}