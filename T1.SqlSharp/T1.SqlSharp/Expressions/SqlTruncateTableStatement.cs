namespace T1.SqlSharp.Expressions;

public class SqlTruncateTableStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.TruncateTableStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_TruncateTableStatement(this);
    }

    public string TableName { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"TRUNCATE TABLE {TableName}";
    }
}
