using System.Linq.Expressions;
using Microsoft.EntityFrameworkCore.Metadata;

namespace T1.EfCore;

public class SqlBuilder
{
    public string CreateColumns(string tableName, List<SqlRawProperty> rowProperties)
    {
        return string.Join(", ", rowProperties.Select(x => $"[{tableName}].[{x.ColumnName}]"));
    }

    public string CreateMatchConditionSql<TEntity>(IEntityType entityType, Expression<Func<TEntity, object>> matchExpression) where TEntity : class
    {
        var matchExpressions = entityType.GetMatchConditionProperties(matchExpression)
            .Select(x => x.Name)
            .ToList();
        return string.Join(" and ", matchExpressions.Select(x => $"target.[{x}] = source.[{x}]"));
    }
}