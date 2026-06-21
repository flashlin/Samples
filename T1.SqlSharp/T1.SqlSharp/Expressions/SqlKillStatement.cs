using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlKillStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.KillStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_KillStatement(this);
    }

    public string SessionId { get; set; } = string.Empty;
    public bool WithStatusOnly { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"KILL {SessionId}");
        if (WithStatusOnly)
        {
            sql.Append(" WITH STATUSONLY");
        }

        return sql.ToString();
    }
}
