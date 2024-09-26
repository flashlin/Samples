using Microsoft.EntityFrameworkCore.Metadata;

namespace T1.EfCore;

public class SqlColumnProperty
{
    public required IProperty Property { get; set; }
    public string ColumnName { get; set; } = string.Empty;
    public bool AllowInsert { get; set; }
}