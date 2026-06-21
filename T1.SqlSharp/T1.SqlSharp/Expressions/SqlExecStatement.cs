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
    public string ReturnVariable { get; set; } = string.Empty;
    public List<ISqlExpression> Arguments { get; set; } = [];
    public ISqlExpression? DynamicSql { get; set; }
    public string AtLinkedServer { get; set; } = string.Empty;

    public string ToSql()
    {
        var atClause = string.IsNullOrEmpty(AtLinkedServer) ? string.Empty : $" AT {AtLinkedServer}";
        if (DynamicSql != null)
        {
            return $"EXEC ({DynamicSql.ToSql()}){atClause}";
        }

        var sql = new StringBuilder();
        var prefix = string.IsNullOrEmpty(ReturnVariable) ? string.Empty : $"{ReturnVariable} = ";
        sql.Append($"EXEC {prefix}{ProcedureName}");
        if (Arguments.Count > 0)
        {
            sql.Append($" {string.Join(", ", Arguments.Select(a => a.ToSql()))}");
        }

        sql.Append(atClause);
        return sql.ToString();
    }
}
