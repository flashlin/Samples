using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata;

namespace T1.EfCore;

public class SqlRawPropertyBuilder
{
    public IEnumerable<List<SqlRawProperty>> CreateSqlRawData<TEntity>(List<IProperty> rowProperties, IEnumerable<TEntity> entities)
    {
        var startArgumentIndex = 0;
        foreach (var entity in entities)
        {
            var entityRawProperties = GetSqlRawProperties(rowProperties, entity).ToList();
            foreach (var sqlRawProperty in entityRawProperties)
            {
                sqlRawProperty.DataValue.ArgumentIndex += startArgumentIndex;
            }
            startArgumentIndex += rowProperties.Count;
            yield return entityRawProperties;
        }
    }

    public IEnumerable<SqlRawProperty> GetSqlRawProperties<TEntity>(List<IProperty> rowProperties, TEntity entity)
    {
        return rowProperties.Select((p, index) => p.GetSqlRawProperty(index, entity));
    }

    public List<SqlColumnProperty> GetSqlColumnProperties(IEntityType entityType)
    {
        return entityType.GetProperties().Select(x =>
            new SqlColumnProperty()
            {
                Property = x,
                ColumnName = x.GetColumnName(),
                AllowInsert = x.IsAllowInsert()
            }).ToList();
    }
    
}