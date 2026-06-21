namespace T1.SqlSharp.Expressions;

public class SqlAlterSchemaStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.AlterSchemaStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterSchemaStatement(this);
    }

    public string SchemaName { get; set; } = string.Empty;
    public string ObjectName { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"ALTER SCHEMA {SchemaName} TRANSFER {ObjectName}";
    }
}
