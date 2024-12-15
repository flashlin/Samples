using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlGroupByClause : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.GroupByClause;
    public TextSpan Span { get; set; } = new();
    public List<ISqlExpression> Columns { get; set; } = [];
    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("GROUP BY ");
        foreach (var column in Columns.Select((value, index) => new { value, index }))
        {
            sql.Write(column.value.ToSql());
            if (column.index < Columns.Count - 1)
            {
                sql.Write(", ");
            }
        }
        return sql.ToString();
    }
}