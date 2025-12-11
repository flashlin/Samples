using System.Linq.Expressions;
using Microsoft.EntityFrameworkCore;
using T1.SqlSharp.Expressions;

namespace T1.SqlSharp.Helper;

public static class SqlSelectExpressionBuilder
{
    public static SqlSelectExpressionBuilder<TEntity> From<TEntity>(DbSet<TEntity> dbSet) where TEntity : class
    {
        var tableName = SqlBuilderHelper.GetTableName(dbSet);
        var selectStatement = CreateSelectStatement(tableName);
        var context = SqlBuilderHelper.CreateContext(dbSet);

        return new SqlSelectExpressionBuilder<TEntity>(selectStatement, context);
    }

    private static SelectStatement CreateSelectStatement(string tableName)
    {
        return new SelectStatement
        {
            FromSources = [new SqlTableSource 
            { 
                TableName = tableName,
                Withs = [new SqlHint { Name = "NOLOCK" }]
            }]
        };
    }
}

public class SqlSelectExpressionBuilder<TEntity> where TEntity : class
{
    private readonly SelectStatement _selectStatement;
    private readonly SqlExpressionBuilderContext _context;

    internal SqlSelectExpressionBuilder(SelectStatement selectStatement, SqlExpressionBuilderContext context)
    {
        _selectStatement = selectStatement;
        _context = context;
    }

    public SqlSelectExpressionBuilder<TEntity> Where(Expression<Func<TEntity, bool>> predicate)
    {
        var visitor = new ExpressionTreeVisitor(_context);
        var whereExpression = visitor.Visit(predicate.Body);
        _selectStatement.Where = whereExpression;
        return this;
    }

    public SqlSelectExpressionBuilder<TEntity> Select()
    {
        if (_selectStatement.Columns.Count == 0)
        {
            PopulateAllColumns();
        }
        return this;
    }

    public SqlSelectExpressionBuilder<TEntity> Select<TProperty>(Expression<Func<TEntity, TProperty>> selector)
    {
        _selectStatement.Columns.Clear();

        var propertyName = ExtractPropertyName(selector);

        var columnExpression = new SqlColumnExpression
        {
            Schema = _context.Schema,
            TableName = _context.TableName,
            ColumnName = propertyName
        };

        var alias = $"{_context.TableName}_{propertyName}";

        _selectStatement.Columns.Add(new SelectColumn
        {
            Field = columnExpression,
            Alias = alias
        });

        return this;
    }

    public SqlSelectExpressionBuilder<TEntity> Distinct()
    {
        _selectStatement.SelectType = SelectType.Distinct;
        return this;
    }

    public SqlSelectExpressionBuilder<TEntity> Take(int count)
    {
        _selectStatement.Top = new SqlTopClause
        {
            Expression = new SqlValue
            {
                SqlType = SqlType.IntValue,
                Value = count.ToString()
            }
        };
        return this;
    }

    public SelectStatement Build()
    {
        if (_selectStatement.Columns.Count == 0)
        {
            PopulateAllColumns();
        }
        return _selectStatement;
    }

    private string ExtractPropertyName<TProperty>(Expression<Func<TEntity, TProperty>> selector)
    {
        if (selector.Body is MemberExpression memberExpression)
        {
            return memberExpression.Member.Name;
        }
        throw new ArgumentException("Selector must be a property expression");
    }

    private void PopulateAllColumns()
    {
        var entityType = typeof(TEntity);

        foreach (var property in entityType.GetProperties())
        {
            var columnExpression = new SqlColumnExpression
            {
                Schema = _context.Schema,
                TableName = _context.TableName,
                ColumnName = property.Name
            };

            var alias = $"{_context.TableName}_{property.Name}";

            _selectStatement.Columns.Add(new SelectColumn
            {
                Field = columnExpression,
                Alias = alias
            });
        }
    }
}

