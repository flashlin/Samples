using Microsoft.EntityFrameworkCore;
using T1.SqlSharp.Expressions;

namespace T1.SqlSharp.Helper;

public class SqlExpressionBuilder
{
    private readonly SelectStatement _selectStatement;

    private SqlExpressionBuilder(SelectStatement selectStatement)
    {
        _selectStatement = selectStatement;
    }

    public static SqlExpressionBuilder From<TEntity>(DbSet<TEntity> dbSet) where TEntity : class
    {
        var tableName = GetTableName(dbSet);
        var selectStatement = CreateSelectStatement(tableName);
        return new SqlExpressionBuilder(selectStatement);
    }

    public SelectStatement Select()
    {
        return _selectStatement;
    }

    private static string GetTableName<TEntity>(DbSet<TEntity> dbSet) where TEntity : class
    {
        var entityType = dbSet.EntityType;
        var schema = entityType.GetSchema();
        var tableName = entityType.GetTableName();

        return FormatTableName(schema, tableName);
    }

    private static string FormatTableName(string? schema, string? tableName)
    {
        if (string.IsNullOrEmpty(schema))
            return $"[{tableName}]";

        return $"[{schema}].[{tableName}]";
    }

    private static SelectStatement CreateSelectStatement(string tableName)
    {
        return new SelectStatement
        {
            FromSources = [new SqlTableSource { TableName = tableName }]
        };
    }
}
