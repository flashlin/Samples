namespace T1.SqlSharp.Expressions;

public class SqlUnaryExpr : ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.UnaryExpression;
    public UnaryOperator Operator { get; set; } = UnaryOperator.BitwiseNot;
    public required ISqlExpression Operand { get; set; }

    public string ToSql()
    {
        return $"{Operator.ToSql()} {Operand.ToSql()}";
    }
}