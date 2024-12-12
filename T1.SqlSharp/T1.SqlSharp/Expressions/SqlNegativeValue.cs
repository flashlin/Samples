namespace T1.SqlSharp.Expressions;

public class SqlNegativeValue : ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.NegativeValue;
    public required ISqlExpression Value { get; set; }

    public string ToSql()
    {
        return $"-{Value.ToSql()}";
    }
}