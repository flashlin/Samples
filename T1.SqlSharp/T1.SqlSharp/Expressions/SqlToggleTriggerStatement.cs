using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlToggleTriggerStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.ToggleTriggerStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ToggleTriggerStatement(this);
    }

    public bool Enable { get; set; }
    public List<string> TriggerNames { get; set; } = [];
    public string Target { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append(Enable ? "ENABLE" : "DISABLE");
        sql.Append($" TRIGGER {string.Join(", ", TriggerNames)} ON {Target}");
        return sql.ToString();
    }
}
