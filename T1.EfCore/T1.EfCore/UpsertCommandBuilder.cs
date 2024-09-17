using System.Data.Common;
using System.Linq.Expressions;
using System.Reflection;
using System.Text;
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
                         ?? (entities.Length == 0
                             ? null
                             : dbContext.GetService<IModel>().FindEntityType(entities.First().GetType()))
                         ?? throw new InvalidOperationException();
        return new UpsertCommandBuilder<TEntity>(dbContext, entityType, entities);
    }
}

public class UpsertCommandBuilder<TEntity> where TEntity : class
{
    private readonly DbContext _dbContext;
    private readonly IEntityType _entityType;
    private readonly TEntity[] _entityList;
    private Expression<Func<TEntity, object>>? _matchExpression;

    public UpsertCommandBuilder(DbContext dbContext, IEntityType entityType, params TEntity[] entities)
    {
        _dbContext = dbContext;
        _entityType = entityType;
        _entityList = entities;
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
        var insertColumns = CreateInsertColumns(sqlGenerator, properties);
        var rawProperties = GetSqlRawProperties(properties, _entityList[0]).ToList();
        var createMemTableSql = CreateMemoryTableSql(rawProperties, insertColumns);
        var sourceColumns = string.Join(", ", rawProperties.Select(x => $"source.[{x.ColumnName}]"));

        if (_matchExpression == null)
        {
            throw new InvalidOperationException("On Method IsRequired");
        }

        var matchExpressions = GenerateMatchCondition(_matchExpression)
            .Select(x => x.Name)
            .ToList();

        var onCondition = string.Join("", matchExpressions.Select(x => $"target.{x} = source.{x}"));

        var mergeSql2 = $@"{createMemTableSql}
MERGE INTO {fullTableName} AS target
USING #TempMemoryTable AS source
ON ({onCondition})
WHEN NOT MATCHED THEN
    INSERT ({insertColumns})
    VALUES ({sourceColumns});";


        using var dbCommand = _dbContext.Database.GetDbConnection().CreateCommand();

        var values = CreateDbParameters(dbCommand, rawProperties)
            .ToList();
        _dbContext.Database.ExecuteSqlRaw(mergeSql2, values);
    }

    private static string CreateInsertColumns(ISqlGenerationHelper sqlGenerator, List<IProperty> properties)
    {
        return string.Join(", ", properties.Select(p => sqlGenerator.DelimitIdentifier(p.GetColumnName())));
    }

    private string CreateMemoryTableSql(List<SqlRawProperty> rawProperties, string insertColumns)
    {
        var createMemTableSql = $"CREATE TABLE #TempMemoryTable ({CreateTableColumnsTypes(rawProperties)});";
        var insertMemTableSql = CreateInsertIntoMemoryTableSql(rawProperties, insertColumns);
        return createMemTableSql + "\n" + insertMemTableSql;
    }

    private string CreateInsertIntoMemoryTableSql(List<SqlRawProperty> rawProperties, string insertColumns)
    {
        var sql = new StringBuilder();
        var startArgumentIndex = 0;
        var properties = rawProperties.Select(p => p.Property).ToList();
        foreach (var entity in _entityList)
        {
            var entityRawProperties = GetSqlRawProperties(properties, entity).ToList();
            var insertIntoMemoryTableValue =
                CreateInsertIntoMemoryTableValueSql(entityRawProperties, insertColumns, startArgumentIndex);
            startArgumentIndex += entityRawProperties.Count;
            sql.AppendLine(insertIntoMemoryTableValue);
        }

        return sql.ToString();
    }

    private static string CreateInsertIntoMemoryTableValueSql(List<SqlRawProperty> rawProperties, string insertColumns,
        int startArgumentIndex)
    {
        var insertValues = CreateInsertValues(startArgumentIndex, rawProperties);
        return $@"INSERT INTO #TempMemoryTable ({insertColumns}) VALUES ({insertValues});";
    }

    private static string CreateInsertValues(int startArgumentIndex, List<SqlRawProperty> rawProperties)
    {
        return string.Join(", ", rawProperties.Select(x => $"@p{x.Value.ArgumentIndex + startArgumentIndex}"));
    }

    private static string CreateTableColumnsTypes(List<SqlRawProperty> rawProperties)
    {
        return string.Join(", ", rawProperties.Select(x => $"[{x.PropertyName}] {x.Value.GetColumnType()}"));
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
        var properties = rawProperties.Select(x => x.Property).ToList();
        var startArgumentIndex = 0;
        foreach (var entity in _entityList)
        {
            var entityRawProperties = GetSqlRawProperties(properties, entity).ToList();
            var dbParameters = entityRawProperties.Select(x =>
            {
                var dbCommandArgumentBuilder = new DbCommandArgumentBuilder(_dbContext, dbCommand);
                return dbCommandArgumentBuilder.CreateDbParameter(startArgumentIndex, x.Value);
            }).ToList();
            foreach (var dbParameter in dbParameters)
            {
                yield return dbParameter;
            }
            startArgumentIndex += properties.Count;
        }
    }

    private IEnumerable<SqlRawProperty> GetSqlRawProperties(List<IProperty> properties, TEntity entity)
    {
        return properties.Select((p, index) => p.GetSqlRawProperty(index, entity));
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
            if (unaryExpression.Operand is not MemberExpression memberExp || memberExp.Member is not PropertyInfo ||
                typeof(TEntity) != memberExp.Expression?.Type)
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