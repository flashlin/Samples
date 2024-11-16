namespace SqlSharpLit.Common.ParserLit;

public class ColumnDefinition : ISqlExpression
{
    public string ColumnName { get; set; } = string.Empty;
    public string DataType { get; set; } = string.Empty;
    public bool IsPrimaryKey { get; set; }
    public bool IsAutoIncrement { get; set; }
    public int Size { get; set; }
    public int Scale { get; set; }
    public SqlIdentity Identity { get; set; } = SqlIdentity.Default;
    public bool IsNullable { get; set; }
    public bool NotForReplication { get; set; }
}