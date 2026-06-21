namespace T1.SqlSharp.Expressions;

public class SqlAlterTableGenericAction : ISqlAlterTableAction
{
    public SqlType SqlType => SqlType.AlterTableGenericAction;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterTableGenericAction(this);
    }

    public string Action { get; set; } = string.Empty;

    public string ToSql()
    {
        return Action;
    }
}
