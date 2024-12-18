namespace T1.SqlSharp.Expressions;

public class SqlFieldExpr : ISqlExpression
{
    public SqlType SqlType => SqlType.Field;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_FieldExpr(this);
    }

    public string FieldName { get; set; } = string.Empty;
    public string ToSql()
    {
        return FieldName;
    }
}