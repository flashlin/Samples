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
    public List<ISqlExpression> Elements { get; set; } = [];

    public string ToSql()
    {
        var sql = $"CREATE SCHEMA {SchemaName}";
        if (!string.IsNullOrEmpty(Authorization))
        {
            sql = $"{sql} AUTHORIZATION {Authorization}";
        }

        if (Elements.Count > 0)
        {
            sql = $"{sql} {string.Join(" ", Elements.Select(element => element.ToSql()))}";
        }

        return sql;
    }
}
