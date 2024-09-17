using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Metadata;

namespace T1.EfCore;

public static class DbContextUpsertExtension
{
    public static UpsertCommandBuilder<TEntity> Upsert<TEntity>(this DbContext dbContext, params TEntity[] entities)
        where TEntity : class
    {
        var entityType = dbContext.GetService<IModel>().FindEntityType(typeof(TEntity))
                         ?? (entities.Length == 0
                             ? null
                             : dbContext.GetService<IModel>().FindEntityType(entities.First().GetType()))
                         ?? throw new InvalidOperationException();
        return new UpsertCommandBuilder<TEntity>(dbContext, entityType, entities);
    }
}