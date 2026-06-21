using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlStatisticsStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.StatisticsStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_StatisticsStatement(this);
    }

    public bool IsCreate { get; set; }
    public string Name { get; set; } = string.Empty;
    public string TableName { get; set; } = string.Empty;
    public List<string> Columns { get; set; } = [];
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        if (IsCreate)
        {
            sql.Append($"CREATE STATISTICS {Name} ON {TableName} ({string.Join(", ", Columns)})");
        }
        else
        {
            sql.Append($"UPDATE STATISTICS {TableName}");
            if (!string.IsNullOrEmpty(Name))
            {
                sql.Append($" {Name}");
            }
        }

        if (Options.Count > 0)
        {
            sql.Append($" WITH {string.Join(", ", Options)}");
        }

        return sql.ToString();
    }
}
