namespace T1.SqlSharp.Expressions;

public class SqlAlterTableToggleTrigger : ISqlAlterTableAction
{
    public SqlType SqlType => SqlType.AlterTableToggleTrigger;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterTableToggleTrigger(this);
    }

    public bool Enable { get; set; }
    public bool AllTriggers { get; set; }
    public List<string> TriggerNames { get; set; } = [];

    public string ToSql()
    {
        var keyword = Enable ? "ENABLE" : "DISABLE";
        var target = AllTriggers ? "ALL" : string.Join(", ", TriggerNames);
        return $"{keyword} TRIGGER {target}";
    }
}
