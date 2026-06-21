using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlGroupByClause : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.GroupByClause;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_GroupByClause(this);
    }

    public GroupingType GroupingType { get; set; } = GroupingType.Simple;
    public bool IsAll { get; set; }
    public List<ISqlExpression> Columns { get; set; } = [];
    public List<SqlGroupingSet> GroupingSets { get; set; } = [];

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("GROUP BY ");
        if (IsAll)
        {
            sql.Write("ALL ");
        }
        sql.Write(GroupingBodyToSql());
        return sql.ToString();
    }

    private string GroupingBodyToSql()
    {
        return GroupingType switch
        {
            GroupingType.Rollup => $"ROLLUP ({ColumnsToSql()})",
            GroupingType.Cube => $"CUBE ({ColumnsToSql()})",
            GroupingType.GroupingSets => $"GROUPING SETS ({string.Join(", ", GroupingSets.Select(set => set.ToSql()))})",
            _ => ColumnsToSql()
        };
    }

    private string ColumnsToSql()
    {
        return string.Join(", ", Columns.Select(column => column.ToSql()));
    }
}