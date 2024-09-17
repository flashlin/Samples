using System.Data.Common;
using System.Linq.Expressions;
using System.Reflection;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Metadata;
using Microsoft.EntityFrameworkCore.Storage;

namespace T1.EfCore;

public static class UpsertExtension
{
    public static UpsertCommandBuilder<TEntity> Upsert<TEntity>(this DbContext dbContext, params TEntity[] entities)
        where TEntity : class
    {
        var entityType = dbContext.GetService<IModel>().FindEntityType(typeof(TEntity))
                         ?? (entities.Length == 0 ? null : dbContext.GetService<IModel>().FindEntityType(entities.First().GetType()))
                         ?? throw new InvalidOperationException();
        return new UpsertCommandBuilder<TEntity>(dbContext, entityType, entities[0]);
    }
}

public class UpsertCommandBuilder<TEntity> where TEntity : class
{
    private readonly DbContext _dbContext;
    private readonly IEntityType _entityType;
    private readonly TEntity _entity;
    private Expression<Func<TEntity, object>> _matchExpression;

    public UpsertCommandBuilder(DbContext dbContext, IEntityType entityType, TEntity entity)
    {
        _dbContext = dbContext;
        _entityType = entityType;
        _entity = entity;
    }

    public UpsertCommandBuilder<TEntity> On(Expression<Func<TEntity, object>> matchExpression)
    {
        _matchExpression = matchExpression;
        return this;
    }

    public void Execute()
    {
        var sqlGenerator = _dbContext.GetService<ISqlGenerationHelper>();
        var fullTableName = GetFullTableName(sqlGenerator);

        var properties = _entityType.GetProperties().ToList();
        var rawProperties = GetSqlRawProperties(properties).ToList();

        var insertColumns =
             string.Join(", ", properties.Select(p => sqlGenerator.DelimitIdentifier(p.GetColumnName())));
        var insertValues = string.Join(", ", rawProperties.Select(x=> $"@p{x.Value.ArgumentIndex}"));
        var matchExpressions = GenerateMatchCondition(_matchExpression)
            .Select(x => x.Name)
            .ToList();
        
        var onCondition = string.Join("", matchExpressions.Select(x => $"target.{x} = source.{x}"));

        var mergeSql = $@"
MERGE INTO {fullTableName} AS target
USING (SELECT {insertValues}) AS source ({insertColumns})
ON ({onCondition})
WHEN NOT MATCHED THEN
    INSERT ({insertColumns})
    VALUES ({insertValues});";

        using var dbCommand = _dbContext.Database.GetDbConnection().CreateCommand();
        
        var values = CreateDbParameters(dbCommand, rawProperties)
            .ToList();
        _dbContext.Database.ExecuteSqlRaw(mergeSql, values);
    }

    private string GetFullTableName(ISqlGenerationHelper sqlGenerator)
    {
        var tableName = _entityType.GetTableName() ??
                        _entityType.GetDefaultTableName() ?? _entityType.GetType().Name;
        var schema = _entityType.GetSchema();
        var fullTableName = sqlGenerator.DelimitIdentifier(tableName, schema);
        return fullTableName;
    }

    private IEnumerable<DbParameter> CreateDbParameters(DbCommand dbCommand, List<SqlRawProperty> rawProperties)
    {
        return rawProperties.Select(x =>
        {
            var dbCommandArgumentBuilder = new DbCommandArgumentBuilder(_dbContext, dbCommand);
            return dbCommandArgumentBuilder.CreateDbParameter(x.Value);
        });
    }

    private IEnumerable<SqlRawProperty> GetSqlRawProperties(List<IProperty> properties)
    {
        return properties.Select((p, index) =>
        {
            var item = p.GetSqlRawProperty(_entity);
            item.Value.ArgumentIndex = index;
            return item;
        });
    }

    private List<IProperty> GenerateMatchCondition(Expression<Func<TEntity, object>> matchExpression)
    {
        if (matchExpression.Body is MemberExpression memberExpression)
        {
            if (typeof(TEntity) != memberExpression.Expression?.Type || memberExpression.Member is not PropertyInfo)
                throw new InvalidOperationException("MatchColumnsHaveToBePropertiesOfTheTEntityClass");
            var property = _entityType.FindProperty(memberExpression.Member.Name);
            if (property == null)
                throw new InvalidOperationException("UnknownProperty memberExpression.Member.Name");
            return [property];
        }
        
        if (matchExpression.Body is UnaryExpression unaryExpression)
        {
            if (unaryExpression.Operand is not MemberExpression memberExp || memberExp.Member is not PropertyInfo || typeof(TEntity) != memberExp.Expression?.Type)
                throw new InvalidOperationException("MatchColumnsHaveToBePropertiesOfTheTEntityClass");
            var property = _entityType.FindProperty(memberExp.Member.Name);
            if (property == null)
                throw new InvalidOperationException("UnknownProperty, memberExp.Member.Name");
            return [property];
        }
        
        
        // if (matchExpression.Body is MemberExpression memberExpression)
        // {
        //     var propertyInfo = (PropertyInfo)memberExpression.Member;
        //     var columnName = _entityType.FindProperty(propertyInfo.Name).GetColumnName();
        //     return $"{sqlGenerator.DelimitIdentifier(columnName)} = @{propertyInfo.Name}";
        // }

        throw new ArgumentException("Unsupported where expression");
    }
}