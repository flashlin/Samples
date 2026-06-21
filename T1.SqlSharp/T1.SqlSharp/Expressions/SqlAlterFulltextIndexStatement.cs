namespace T1.SqlSharp.Expressions;

public class SqlAlterFulltextIndexStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.AlterFulltextIndexStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterFulltextIndexStatement(this);
    }

    public string TableName { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"ALTER FULLTEXT INDEX ON {TableName} {Action}";
    }
}
