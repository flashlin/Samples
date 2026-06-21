using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateFulltextIndexStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateFulltextIndexStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateFulltextIndexStatement(this);
    }

    public string TableName { get; set; } = string.Empty;
    public List<string> Columns { get; set; } = [];
    public string KeyIndex { get; set; } = string.Empty;
    public string Catalog { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE FULLTEXT INDEX ON {TableName} ({string.Join(", ", Columns)})");
        if (!string.IsNullOrEmpty(KeyIndex))
        {
            sql.Append($" KEY INDEX {KeyIndex}");
        }

        if (!string.IsNullOrEmpty(Catalog))
        {
            sql.Append($" ON {Catalog}");
        }

        return sql.ToString();
    }
}
