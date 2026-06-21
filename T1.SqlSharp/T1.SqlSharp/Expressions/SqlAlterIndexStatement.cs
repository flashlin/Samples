namespace T1.SqlSharp.Expressions;

public class SqlAlterIndexStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.AlterIndexStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterIndexStatement(this);
    }

    public string IndexName { get; set; } = string.Empty;
    public string TableName { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"ALTER INDEX {IndexName} ON {TableName} {Action}";
    }
}
