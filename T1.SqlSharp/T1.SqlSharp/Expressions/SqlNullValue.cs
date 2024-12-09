namespace T1.SqlSharp.Expressions;

public class SqlNullValue : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.Null;
    public string ToSql()
    {
        return "NULL";
    }
}