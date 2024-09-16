namespace T1.EfCore;

public class SqlRawProperty
{
    public string PropertyName { get; set; } = string.Empty;
    public string ColumnName { get; set; }  = string.Empty;
    public ConstantValue Value { get; set; }
    public string? DefaultSql { get; set; }
    public bool AllowInsert { get; set; }
}