namespace T1.SqlSharp.Expressions;

public class SqlCaseWhenExpression : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.CaseWhen;
    public required ISqlExpression WhenExpr { get; set; }
    public required ISqlExpression Then { get; set; }

    public string ToSql()
    {
        var sql = "WHEN " + WhenExpr.ToSql() + " THEN " + Then.ToSql();
        return sql;
    }
}