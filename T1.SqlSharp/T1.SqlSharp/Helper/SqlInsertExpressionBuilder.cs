using Microsoft.EntityFrameworkCore;
using T1.SqlSharp.Expressions;

namespace T1.SqlSharp.Helper;

public static class SqlInsertExpressionBuilder
{
    public static SqlInsertExpressionBuilder<TEntity> Into<TEntity>(DbSet<TEntity> dbSet) where TEntity : class
    {
        var tableName = SqlBuilderHelper.GetTableName(dbSet);
        var columns = SqlBuilderHelper.GetEntityPropertyNames<TEntity>();
        var insertStatement = CreateInsertStatement(tableName, columns);
        var context = SqlBuilderHelper.CreateContext(dbSet);

        return new SqlInsertExpressionBuilder<TEntity>(insertStatement, context);
    }

    private static SqlInsertStatement CreateInsertStatement(string tableName, List<string> columns)
    {
        return new SqlInsertStatement
        {
            TableName = tableName,
            Columns = columns
        };
    }
}

public class SqlInsertExpressionBuilder<TEntity> where TEntity : class
{
    private readonly SqlInsertStatement _insertStatement;
    private readonly SqlExpressionBuilderContext _context;

    internal SqlInsertExpressionBuilder(SqlInsertStatement insertStatement, SqlExpressionBuilderContext context)
    {
        _insertStatement = insertStatement;
        _context = context;
    }

    public SqlInsertStatement Build()
    {
        return _insertStatement;
    }
}

