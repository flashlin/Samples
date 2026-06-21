namespace T1.SqlSharp.Expressions;

public class SqlCreateSchemaStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateSchemaStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateSchemaStatement(this);
    }

    public string SchemaName { get; set; } = string.Empty;
    public string Authorization { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = $"CREATE SCHEMA {SchemaName}";
        return string.IsNullOrEmpty(Authorization) ? sql : $"{sql} AUTHORIZATION {Authorization}";
    }
}
