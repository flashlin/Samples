namespace T1.SqlSharp.Expressions;

public class SqlAlterTableColumnOption : ISqlAlterTableAction
{
    public SqlType SqlType => SqlType.AlterTableColumnOption;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterTableColumnOption(this);
    }

    public string ColumnName { get; set; } = string.Empty;
    public bool IsAdd { get; set; }
    public string Option { get; set; } = string.Empty;

    public string ToSql()
    {
        var action = IsAdd ? "ADD" : "DROP";
        return $"ALTER COLUMN {ColumnName} {action} {Option}";
    }
}
