namespace T1.SqlSharp.Expressions;

public class SqlFieldExpr : ISqlExpression
{
    public SqlType SqlType => SqlType.Field;
    public TextSpan Span { get; set; } = new();
    public string FieldName { get; set; } = string.Empty;
    public string ToSql()
    {
        return FieldName;
    }
}