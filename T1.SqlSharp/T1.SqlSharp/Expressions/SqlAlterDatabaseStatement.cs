namespace T1.SqlSharp.Expressions;

public class SqlAlterDatabaseStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.AlterDatabaseStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_AlterDatabaseStatement(this);
    }

    public string DatabaseName { get; set; } = string.Empty;
    public string Setting { get; set; } = string.Empty;
    public string SettingValue { get; set; } = string.Empty;
    public string FileAction { get; set; } = string.Empty;
    public string FileSpec { get; set; } = string.Empty;

    public string ToSql()
    {
        if (!string.IsNullOrEmpty(FileAction))
        {
            return $"ALTER DATABASE {DatabaseName} {FileAction} {FileSpec}";
        }

        var sql = $"ALTER DATABASE {DatabaseName} SET {Setting}";
        return string.IsNullOrEmpty(SettingValue) ? sql : $"{sql} {SettingValue}";
    }
}
