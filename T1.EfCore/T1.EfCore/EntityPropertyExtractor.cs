using Microsoft.EntityFrameworkCore.Metadata;

namespace T1.EfCore;

public class EntityPropertyExtractor
{
    public IEnumerable<List<SqlRawProperty>> CreateDataSqlRawProperties<TEntity>(List<IProperty> properties, IEnumerable<TEntity> entities)
    {
        var startArgumentIndex = 0;
        foreach (var entity in entities)
        {
            var entityRawProperties = GetSqlRawProperties(properties, entity).ToList();
            foreach (var sqlRawProperty in entityRawProperties)
            {
                sqlRawProperty.DataValue.ArgumentIndex += startArgumentIndex;
            }
            startArgumentIndex += properties.Count;
            yield return entityRawProperties;
        }
    }

    public IEnumerable<SqlRawProperty> GetSqlRawProperties<TEntity>(List<IProperty> properties, TEntity entity)
    {
        return properties.Select((p, index) => p.GetSqlRawProperty(index, entity));
    }
}