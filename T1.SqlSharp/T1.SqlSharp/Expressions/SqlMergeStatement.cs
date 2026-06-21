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

    public required ITableSource Target { get; set; }
    public required ITableSource Source { get; set; }
    public required ISqlExpression OnCondition { get; set; }
    public List<SqlMergeWhenClause> WhenClauses { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"MERGE INTO {Target.ToSql()} USING {Source.ToSql()} ON {OnCondition.ToSql()}");
        foreach (var whenClause in WhenClauses)
        {
            sql.Append($" {whenClause.ToSql()}");
        }
        return sql.ToString();
    }
}
