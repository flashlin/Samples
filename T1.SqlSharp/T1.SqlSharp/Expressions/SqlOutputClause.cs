using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlOutputClause : ISqlExpression
{
    public SqlType SqlType => SqlType.OutputClause;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_OutputClause(this);
    }

    public List<ISelectColumnExpression> Columns { get; set; } = [];
    public string IntoTable { get; set; } = string.Empty;
    public List<string> IntoColumns { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("OUTPUT ");
        sql.Append(string.Join(", ", Columns.Select(c => c.ToSql())));
        if (!string.IsNullOrEmpty(IntoTable))
        {
            sql.Append($" INTO {IntoTable}");
            if (IntoColumns.Count > 0)
            {
                sql.Append($" ({string.Join(", ", IntoColumns)})");
            }
        }
        return sql.ToString();
    }
}
