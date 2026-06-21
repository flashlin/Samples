namespace T1.SqlSharp.Expressions;

public class SqlAlterTableStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.AlterTableStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterTableStatement(this);
    }

    public string TableName { get; set; } = string.Empty;
    public required ISqlAlterTableAction Action { get; set; }

    public string ToSql()
    {
        return $"ALTER TABLE {TableName} {Action.ToSql()}";
    }
}
