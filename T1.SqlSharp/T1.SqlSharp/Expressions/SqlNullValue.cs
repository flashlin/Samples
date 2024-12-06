namespace T1.SqlSharp.Expressions;

public class SqlNullValue : ISqlValue
{
    public SqlType SqlType { get; } = SqlType.Null;
    public string ToSql()
    {
        return "NULL";
    }
    public string Value { get; } = "NULL"; 
}