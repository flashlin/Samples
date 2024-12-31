namespace SqlSharpLit.Common.ParserLit;

public class SqlCreateTablesSqlFiles
{
    public SqlFileContent File { get; set; } = SqlFileContent.Empty;
    public List<string> CreateTables { get; set; } = [];
    public string DatabaseName { get; set; } = string.Empty;
}