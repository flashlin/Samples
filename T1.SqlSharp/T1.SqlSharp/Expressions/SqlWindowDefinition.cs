using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlWindowDefinition : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.WindowDefinition;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_WindowDefinition(this);
    }

    public string Name { get; set; } = string.Empty;
    public List<ISqlExpression> PartitionBy { get; set; } = [];
    public List<SqlOrderColumn> OrderColumns { get; set; } = [];
    public SqlWindowFrameClause? Frame { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write($"{Name} AS (");
        sql.Write(string.Join(" ", GetSpecParts()));
        sql.Write(")");
        return sql.ToString();
    }

    private List<string> GetSpecParts()
    {
        var parts = new List<string>();
        if (PartitionBy.Count > 0)
        {
            parts.Add("PARTITION BY " + string.Join(", ", PartitionBy.Select(x => x.ToSql())));
        }

        if (OrderColumns.Count > 0)
        {
            parts.Add("ORDER BY " + string.Join(", ", OrderColumns.Select(x => x.ToSql())));
        }

        if (Frame != null)
        {
            parts.Add(Frame.ToSql());
        }

        return parts;
    }
}
