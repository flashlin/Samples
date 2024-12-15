using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlTopClause : ISqlExpression
{
    public SqlType SqlType => SqlType.TopClause;
    public TextSpan Span { get; set; } = new();
    public required ISqlExpression Expression { get; set; }
    public bool IsPercent { get; set; }
    public bool IsWithTies { get; set; }
    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"TOP {Expression.ToSql()}");
        if (IsPercent)
        {
            sql.Append(" PERCENT");
        }
        if(IsWithTies)
        {
            sql.Append(" WITH TIES");
        }
        return sql.ToString();
    }
}