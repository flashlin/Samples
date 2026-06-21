using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlRaiseErrorStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.RaiseErrorStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_RaiseErrorStatement(this);
    }

    public required ISqlExpression Message { get; set; }
    public required ISqlExpression Severity { get; set; }
    public required ISqlExpression State { get; set; }
    public List<ISqlExpression> Arguments { get; set; } = [];
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("RAISERROR (");
        var parts = new List<ISqlExpression> { Message, Severity, State };
        parts.AddRange(Arguments);
        sql.Append(string.Join(", ", parts.Select(x => x.ToSql())));
        sql.Append(')');
        if (Options.Count > 0)
        {
            sql.Append(" WITH ");
            sql.Append(string.Join(", ", Options));
        }
        return sql.ToString();
    }
}
