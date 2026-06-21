using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlDeleteStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.DeleteStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_DeleteStatement(this);
    }

    public SqlTopClause? Top { get; set; }
    public string TableName { get; set; } = string.Empty;
    public List<ISqlExpression> Withs { get; set; } = [];
    public SqlOutputClause? Output { get; set; }
    public List<ISqlExpression> FromSources { get; set; } = [];
    public ISqlExpression? Where { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("DELETE ");
        if (Top != null)
        {
            sql.Append($"{Top.ToSql()} ");
        }
        sql.Append($"FROM {TableName}");
        if (Where != null)
        {
            sql.Append($" WHERE {Where.ToSql()}");
        }
        return sql.ToString();
    }
}
