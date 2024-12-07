namespace T1.SqlSharp.Expressions;

public class SqlAsExpr : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.AsExpr;
    public required ISqlExpression Value { get; set; }
    public required SqlDataType DataType { get; set; }

    public string ToSql()
    {
        throw new NotImplementedException();
    }
}