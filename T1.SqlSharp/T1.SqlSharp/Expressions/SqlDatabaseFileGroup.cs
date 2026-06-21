namespace T1.SqlSharp.Expressions;

public class SqlDatabaseFileGroup
{
    public string Name { get; set; } = string.Empty;
    public List<string> Files { get; set; } = [];

    public string ToSql()
    {
        return $"FILEGROUP {Name} {string.Join(", ", Files)}";
    }
}
