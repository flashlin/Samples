using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlAlterTableSwitch : ISqlAlterTableAction
{
    public SqlType SqlType => SqlType.AlterTableSwitch;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterTableSwitch(this);
    }

    public string SourcePartition { get; set; } = string.Empty;
    public string TargetTable { get; set; } = string.Empty;
    public string TargetPartition { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("SWITCH");
        if (!string.IsNullOrEmpty(SourcePartition))
        {
            sql.Append($" PARTITION {SourcePartition}");
        }

        sql.Append($" TO {TargetTable}");
        if (!string.IsNullOrEmpty(TargetPartition))
        {
            sql.Append($" PARTITION {TargetPartition}");
        }

        return sql.ToString();
    }
}
