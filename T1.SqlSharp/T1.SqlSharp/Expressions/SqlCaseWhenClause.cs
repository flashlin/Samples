namespace T1.SqlSharp.Expressions;

public class SqlCaseWhenClause : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.CaseWhen;
    public required ISqlExpression When { get; set; }
    public required ISqlExpression Then { get; set; }

    public string ToSql()
    {
        var sql = "WHEN " + When.ToSql() + " THEN " + Then.ToSql();
        return sql;
    }
}