using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlConversationStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.ConversationStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ConversationStatement(this);
    }

    public string Operation { get; set; } = string.Empty;
    public string Handle { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append(Operation);
        if (!string.IsNullOrEmpty(Handle))
        {
            sql.Append($" {Handle}");
        }

        if (!string.IsNullOrEmpty(Action))
        {
            sql.Append($" {Action}");
        }

        return sql.ToString();
    }
}
