using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Metadata;

namespace T1.EfCore;

public static class DbContextExtensions
{
    public static UpsertCommandBuilder<TEntity> Upsert<TEntity>(this DbContext dbContext, params TEntity[] entities)
        where TEntity : class
    {
        var entityType = dbContext.GetEntityType(entities[0]);
        return new UpsertCommandBuilder<TEntity>(dbContext, entityType, entities);
    }

    public static IEntityType GetEntityType<TEntity>(this DbContext dbContext, TEntity entity) 
        where TEntity : class
    {
        var entityType = dbContext.GetService<IModel>().FindEntityType(typeof(TEntity))
                         ?? dbContext.GetService<IModel>().FindEntityType(entity.GetType())
                         ?? throw new InvalidOperationException();
        return entityType;
    }

    public static BulkInserter<TEntity> BulkInsert<TEntity>(this DbContext dbContext, IEnumerable<TEntity> entities)
        where TEntity : class
    {
        return new BulkInserter<TEntity>(dbContext, entities); 
    }
}