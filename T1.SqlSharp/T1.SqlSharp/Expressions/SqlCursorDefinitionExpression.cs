using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCursorDefinitionExpression : ISqlExpression
{
    public SqlType SqlType => SqlType.CursorDefinitionExpression;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CursorDefinitionExpression(this);
    }

    public List<string> Options { get; set; } = [];
    public required ISqlExpression Source { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("CURSOR");
        if (Options.Count > 0)
        {
            sql.Append($" {string.Join(" ", Options)}");
        }

        sql.Append($" FOR {Source.ToSql()}");
        return sql.ToString();
    }
}
