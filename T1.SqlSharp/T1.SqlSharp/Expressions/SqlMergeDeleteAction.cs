namespace T1.SqlSharp.Expressions;

public class SqlMergeDeleteAction : ISqlMergeAction
{
    public SqlType SqlType => SqlType.MergeDeleteAction;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_MergeDeleteAction(this);
    }

    public string ToSql()
    {
        return "DELETE";
    }
}
