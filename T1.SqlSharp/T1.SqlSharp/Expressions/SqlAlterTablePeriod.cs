namespace T1.SqlSharp.Expressions;

public class SqlAlterTablePeriod : ISqlAlterTableAction
{
    public SqlType SqlType => SqlType.AlterTablePeriod;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterTablePeriod(this);
    }

    public bool IsAdd { get; set; }
    public List<string> Columns { get; set; } = [];

    public string ToSql()
    {
        var verb = IsAdd ? "ADD" : "DROP";
        if (!IsAdd)
        {
            return $"{verb} PERIOD FOR SYSTEM_TIME";
        }

        return $"{verb} PERIOD FOR SYSTEM_TIME ({string.Join(", ", Columns)})";
    }
}
