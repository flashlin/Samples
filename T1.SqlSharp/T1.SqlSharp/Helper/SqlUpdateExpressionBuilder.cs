using System.Linq.Expressions;
using Microsoft.EntityFrameworkCore;
using T1.SqlSharp.Expressions;

namespace T1.SqlSharp.Helper;

public static class SqlUpdateExpressionBuilder
{
    public static SqlUpdateExpressionBuilder<TEntity> Update<TEntity>(DbSet<TEntity> dbSet) where TEntity : class
    {
        var tableName = SqlBuilderHelper.GetTableName(dbSet);
        var updateStatement = CreateUpdateStatement(tableName);
        var context = SqlBuilderHelper.CreateContext(dbSet);

        return new SqlUpdateExpressionBuilder<TEntity>(updateStatement, context);
    }

    private static SqlUpdateStatement CreateUpdateStatement(string tableName)
    {
        return new SqlUpdateStatement
        {
            TableName = tableName
        };
    }
}

public class SqlUpdateExpressionBuilder<TEntity> where TEntity : class
{
    private readonly SqlUpdateStatement _updateStatement;
    private readonly SqlExpressionBuilderContext _context;
    private int _parameterIndex;

    internal SqlUpdateExpressionBuilder(SqlUpdateStatement updateStatement, SqlExpressionBuilderContext context)
    {
        _updateStatement = updateStatement;
        _context = context;
        _parameterIndex = 0;
    }

    public SqlUpdateExpressionBuilder<TEntity> Set<TProperty>(Expression<Func<TEntity, TProperty>> selector, TProperty value)
    {
        var columnName = ExtractPropertyName(selector);
        var parameterName = $"@p{_parameterIndex++}";

        _updateStatement.SetColumns.Add(new SqlSetColumn
        {
            ColumnName = columnName,
            ParameterName = parameterName,
            Value = value
        });

        return this;
    }

    public SqlUpdateStatement Build()
    {
        return _updateStatement;
    }

    private string ExtractPropertyName<TProperty>(Expression<Func<TEntity, TProperty>> selector)
    {
        if (selector.Body is MemberExpression memberExpression)
        {
            return memberExpression.Member.Name;
        }
        throw new ArgumentException("Selector must be a property expression");
    }
}

