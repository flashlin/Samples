using System.Linq.Expressions;
using Microsoft.EntityFrameworkCore;
using T1.SqlSharp.Expressions;

namespace T1.SqlSharp.Helper;

public static class SqlExpressionBuilder
{
    public static SqlExpressionBuilder<TEntity> From<TEntity>(DbSet<TEntity> dbSet) where TEntity : class
    {
        var tableName = GetTableName(dbSet);
        var selectStatement = CreateSelectStatement(tableName);

        var context = new SqlExpressionBuilderContext
        {
            Schema = dbSet.EntityType.GetSchema() ?? string.Empty,
            TableName = dbSet.EntityType.GetTableName() ?? string.Empty
        };

        return new SqlExpressionBuilder<TEntity>(selectStatement, context);
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

public class SqlExpressionBuilder<TEntity> where TEntity : class
{
    private readonly SelectStatement _selectStatement;
    private readonly SqlExpressionBuilderContext _context;

    internal SqlExpressionBuilder(SelectStatement selectStatement, SqlExpressionBuilderContext context)
    {
        _selectStatement = selectStatement;
        _context = context;
    }

    public SqlExpressionBuilder<TEntity> Where(Expression<Func<TEntity, bool>> predicate)
    {
        var visitor = new ExpressionTreeVisitor(_context);
        var whereExpression = visitor.Visit(predicate.Body);
        _selectStatement.Where = whereExpression;
        return this;
    }

    public SelectStatement Select()
    {
        return _selectStatement;
    }
}
