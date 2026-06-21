using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlDbccStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.DbccStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_DbccStatement(this);
    }

    public string Command { get; set; } = string.Empty;
    public List<string> Arguments { get; set; } = [];
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"DBCC {Command}");
        if (Arguments.Count > 0)
        {
            sql.Append($" ({string.Join(", ", Arguments)})");
        }

        if (Options.Count > 0)
        {
            sql.Append($" WITH {string.Join(", ", Options)}");
        }

        return sql.ToString();
    }
}
