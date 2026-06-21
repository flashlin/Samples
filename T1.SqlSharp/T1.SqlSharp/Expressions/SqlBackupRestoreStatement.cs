using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlBackupRestoreStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.BackupRestoreStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_BackupRestoreStatement(this);
    }

    public bool IsBackup { get; set; }
    public string ObjectKind { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public List<string> Devices { get; set; } = [];
    public List<string> Options { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append(IsBackup ? "BACKUP" : "RESTORE");
        sql.Append($" {ObjectKind} {Name}");
        sql.Append(IsBackup ? " TO " : " FROM ");
        sql.Append(string.Join(", ", Devices));
        if (Options.Count > 0)
        {
            sql.Append($" WITH {string.Join(", ", Options)}");
        }

        return sql.ToString();
    }
}
