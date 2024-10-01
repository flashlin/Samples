using Microsoft.EntityFrameworkCore.Metadata;

namespace T1.EfCore;

public class SqlRawProperty
{
    public required IProperty Property { get; set; }
    public string PropertyName { get; set; } = string.Empty;
    public string ColumnName { get; set; }  = string.Empty;
    public required ConstantValue DataValue { get; set; }
    public string? DefaultSql { get; set; }
    public bool AllowInsert { get; set; }
}
