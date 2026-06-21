using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlSendStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.SendStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_SendStatement(this);
    }

    public string ConversationHandle { get; set; } = string.Empty;
    public string MessageType { get; set; } = string.Empty;
    public string Body { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"SEND ON CONVERSATION {ConversationHandle}");
        if (!string.IsNullOrEmpty(MessageType))
        {
            sql.Append($" MESSAGE TYPE {MessageType}");
        }

        if (!string.IsNullOrEmpty(Body))
        {
            sql.Append($" ({Body})");
        }

        return sql.ToString();
    }
}
