using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlInsertStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.InsertStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_InsertStatement(this);
    }

    public string TableName { get; set; } = string.Empty;
    public List<string> Columns { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"INSERT INTO {TableName} (");
        sql.Append(string.Join(", ", Columns.Select(c => $"[{c}]")));
        sql.Append(") VALUES (");
        sql.Append(string.Join(", ", Columns.Select((_, i) => $"@p{i}")));
        sql.Append(")");
        return sql.ToString();
    }
}

