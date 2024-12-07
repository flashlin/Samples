namespace T1.SqlSharp.Expressions;

public class SqlAsExpr : ISqlValue
{
    public SqlType SqlType { get; } = SqlType.AsExpr;
    public required ISqlExpression Instance { get; set; }
    public required SqlDataType DataType { get; set; }

    public string Value => ToSql();

    public string ToSql()
    {
        return $"{Instance.ToSql()} as {DataType.ToSql()}";
    }
}