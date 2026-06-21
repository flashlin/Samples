using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateFulltextCatalogStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateFulltextCatalogStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateFulltextCatalogStatement(this);
    }

    public string Name { get; set; } = string.Empty;
    public bool IsDefault { get; set; }
    public string Authorization { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE FULLTEXT CATALOG {Name}");
        if (IsDefault)
        {
            sql.Append(" AS DEFAULT");
        }

        if (!string.IsNullOrEmpty(Authorization))
        {
            sql.Append($" AUTHORIZATION {Authorization}");
        }

        return sql.ToString();
    }
}
