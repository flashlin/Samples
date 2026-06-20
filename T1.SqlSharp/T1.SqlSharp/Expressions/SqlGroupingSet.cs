namespace T1.SqlSharp.Expressions;

public class SqlGroupingSet : ISqlExpression
{
    public SqlType SqlType => SqlType.GroupingSet;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_GroupingSet(this);
    }

    public List<ISqlExpression> Columns { get; set; } = [];

    public string ToSql()
    {
        return $"({string.Join(", ", Columns.Select(column => column.ToSql()))})";
    }
}
