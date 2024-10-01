using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata;
using Microsoft.EntityFrameworkCore.Storage;

namespace T1.EfCore;

public static class SqlGenerationHelperExtensions
{
    public static string GetFullTableName(this ISqlGenerationHelper sqlGenerator, IEntityType entityType)
    {
        var tableName = entityType.GetTableName() ??
                        entityType.GetDefaultTableName() ?? entityType.GetType().Name;
        var schema = entityType.GetSchema();
        var fullTableName = sqlGenerator.DelimitIdentifier(tableName, schema);
        return fullTableName;
    }

    public static string CreateSourceColumns(this List<SqlRawProperty> sqlRawProperties)
    {
        return string.Join(", ", sqlRawProperties.Select(x => $"source.[{x.ColumnName}]"));
    }
}