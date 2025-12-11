namespace T1.SqlSharp.Expressions;

public class SqlColumnExpression : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.Field;
    public TextSpan Span { get; set; } = new();

    public string Schema { get; set; } = string.Empty;
    public string TableName { get; set; } = string.Empty;
    public string ColumnName { get; set; } = string.Empty;

    public void Accept(SqlVisitor visitor) { }

    public string ToSql()
    {
        if (!string.IsNullOrEmpty(Schema))
            return $"[{Schema}].[{TableName}].[{ColumnName}]";
        return $"[{TableName}].[{ColumnName}]";
    }
}
