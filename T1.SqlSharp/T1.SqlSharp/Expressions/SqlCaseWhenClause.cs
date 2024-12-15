namespace T1.SqlSharp.Expressions;

public class SqlWhenThenClause : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.WhenThen;
    public TextSpan Span { get; set; } = new();
    public required ISqlExpression When { get; set; }
    public required ISqlExpression Then { get; set; }

    public string ToSql()
    {
        var sql = "WHEN " + When.ToSql() + " THEN " + Then.ToSql();
        return sql;
    }
}