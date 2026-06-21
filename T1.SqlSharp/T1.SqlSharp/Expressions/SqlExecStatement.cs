using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlExecStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.ExecStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ExecStatement(this);
    }

    public string ProcedureName { get; set; } = string.Empty;
    public List<ISqlExpression> Arguments { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"EXEC {ProcedureName}");
        if (Arguments.Count > 0)
        {
            sql.Append($" {string.Join(", ", Arguments.Select(a => a.ToSql()))}");
        }
        return sql.ToString();
    }
}
