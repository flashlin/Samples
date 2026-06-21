namespace T1.SqlSharp.Expressions;

public class SqlReceiveStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.ReceiveStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ReceiveStatement(this);
    }

    public string Top { get; set; } = string.Empty;
    public string FromQueue { get; set; } = string.Empty;

    public string ToSql()
    {
        var top = string.IsNullOrEmpty(Top) ? string.Empty : $"TOP ({Top}) ";
        return $"RECEIVE {top}* FROM {FromQueue}";
    }
}
