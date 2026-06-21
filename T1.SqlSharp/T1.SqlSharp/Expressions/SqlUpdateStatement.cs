using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlUpdateStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.UpdateStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_UpdateStatement(this);
    }

    public SqlTopClause? Top { get; set; }
    public string TableName { get; set; } = string.Empty;
    public List<ISqlExpression> Withs { get; set; } = [];
    public List<SqlAssignExpr> SetClauses { get; set; } = [];
    public SqlOutputClause? Output { get; set; }
    public List<ISqlExpression> FromSources { get; set; } = [];
    public ISqlExpression? Where { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("UPDATE");
        if (Top != null)
        {
            sql.Append($" {Top.ToSql()}");
        }

        sql.Append($" {TableName}");
        if (Withs.Count > 0)
        {
            sql.Append($" WITH ({string.Join(", ", Withs.Select(hint => hint.ToSql()))})");
        }

        sql.Append($" SET {string.Join(", ", SetClauses.Select(clause => clause.ToSql()))}");
        if (Output != null)
        {
            sql.Append($" {Output.ToSql()}");
        }

        if (FromSources.Count > 0)
        {
            sql.Append($" FROM {string.Join(", ", FromSources.Select(source => source.ToSql()))}");
        }

        if (Where != null)
        {
            sql.Append($" WHERE {Where.ToSql()}");
        }

        return sql.ToString();
    }
}

