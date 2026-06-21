using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlFetchStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.FetchStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_FetchStatement(this);
    }

    public string Direction { get; set; } = string.Empty;
    public ISqlExpression? RowCount { get; set; }
    public string CursorName { get; set; } = string.Empty;
    public List<string> IntoVariables { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("FETCH");
        if (!string.IsNullOrEmpty(Direction))
        {
            sql.Append($" {Direction}");
        }

        if (RowCount != null)
        {
            sql.Append($" {RowCount.ToSql()}");
        }

        sql.Append($" FROM {CursorName}");
        if (IntoVariables.Count > 0)
        {
            sql.Append($" INTO {string.Join(", ", IntoVariables)}");
        }

        return sql.ToString();
    }
}
