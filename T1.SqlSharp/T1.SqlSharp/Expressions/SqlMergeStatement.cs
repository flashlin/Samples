using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlMergeStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.MergeStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_MergeStatement(this);
    }

    public SqlTopClause? Top { get; set; }
    public required ITableSource Target { get; set; }
    public required ITableSource Source { get; set; }
    public required ISqlExpression OnCondition { get; set; }
    public List<SqlMergeWhenClause> WhenClauses { get; set; } = [];
    public SqlOutputClause? Output { get; set; }
    public SqlOptionClause? Option { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("MERGE ");
        if (Top != null)
        {
            sql.Append($"{Top.ToSql()} ");
        }
        sql.Append($"INTO {Target.ToSql()} USING {Source.ToSql()} ON {OnCondition.ToSql()}");
        foreach (var whenClause in WhenClauses)
        {
            sql.Append($" {whenClause.ToSql()}");
        }
        if (Output != null)
        {
            sql.Append($" {Output.ToSql()}");
        }
        if (Option != null)
        {
            sql.Append($" {Option.ToSql()}");
        }
        return sql.ToString();
    }
}
