using Microsoft.EntityFrameworkCore;

namespace T1.SqlSharp.Helper;

public static class SqlBuilderHelper
{
    public static string GetTableName<TEntity>(DbSet<TEntity> dbSet) where TEntity : class
    {
        var entityType = dbSet.EntityType;
        var schema = entityType.GetSchema();
        var tableName = entityType.GetTableName();

        return FormatTableName(schema, tableName);
    }

    public static string FormatTableName(string? schema, string? tableName)
    {
        if (string.IsNullOrEmpty(schema))
            return $"[{tableName}]";

        return $"[{schema}].[{tableName}]";
    }

    public static SqlExpressionBuilderContext CreateContext<TEntity>(DbSet<TEntity> dbSet) where TEntity : class
    {
        return new SqlExpressionBuilderContext
        {
            Schema = dbSet.EntityType.GetSchema() ?? string.Empty,
            TableName = dbSet.EntityType.GetTableName() ?? string.Empty
        };
    }

    public static List<string> GetEntityPropertyNames<TEntity>() where TEntity : class
    {
        var entityType = typeof(TEntity);
        return entityType.GetProperties().Select(p => p.Name).ToList();
    }
}

