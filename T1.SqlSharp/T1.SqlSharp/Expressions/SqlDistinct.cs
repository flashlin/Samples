namespace T1.SqlSharp.Expressions;

public class SqlDistinct : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.Distinct;
    public required ISqlExpression Value { get; set; }

    public string ToSql()
    {
        return $"DISTINCT {Value.ToSql()}";
    }
}