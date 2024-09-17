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
    private readonly TEntity[] _entityList;
    private readonly IEntityType _entityType;
    private Expression<Func<TEntity, object>>? _matchExpression;

    public UpsertCommandBuilder(DbContext dbContext, IEntityType entityType, params TEntity[] entities)
    {
        _dbContext = dbContext;
        _entityType = entityType;
        _entityList = entities;
    }

    public void Execute()
    {
        var sqlGenerator = _dbContext.GetService<ISqlGenerationHelper>();
        var fullTableName = GetFullTableName(sqlGenerator);

        var properties = _entityType.GetProperties().ToList();
        var dataSqlRawProperties = CreateDataSqlRawProperties(properties).ToList();
        var insertColumns = CreateInsertColumns(sqlGenerator, properties);
        var createMemTableSql = CreateMemoryTableSql(insertColumns, dataSqlRawProperties);
        var sourceColumns = CreateSourceColumns(dataSqlRawProperties);

        var matchCondition = CreateMatchCondition();

        var mergeSql2 = $@"{createMemTableSql}
MERGE INTO {fullTableName} AS target
USING #TempMemoryTable AS source
ON ({matchCondition})
WHEN NOT MATCHED THEN
    INSERT ({insertColumns})
    VALUES ({sourceColumns});";


        using var dbCommand = _dbContext.Database.GetDbConnection().CreateCommand();

        var values = CreateDbParameters(dbCommand, dataSqlRawProperties[0])
            .ToList();
        _dbContext.Database.ExecuteSqlRaw(mergeSql2, values);
    }

    public UpsertCommandBuilder<TEntity> On(Expression<Func<TEntity, object>> matchExpression)
    {
        _matchExpression = matchExpression;
        return this;
    }


    private IEnumerable<List<SqlRawProperty>> CreateDataSqlRawProperties(List<IProperty> properties)
    {
        var startArgumentIndex = 0;
        foreach (var entity in _entityList)
        {
            var entityRawProperties = GetSqlRawProperties(properties, entity).ToList();
            foreach (var sqlRawProperty in entityRawProperties)
            {
                sqlRawProperty.Value.ArgumentIndex += startArgumentIndex;
            }
            startArgumentIndex += properties.Count;
            yield return entityRawProperties;
        }
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

    private static string CreateInsertColumns(ISqlGenerationHelper sqlGenerator, List<IProperty> properties)
    {
        return string.Join(", ", properties.Select(p => sqlGenerator.DelimitIdentifier(p.GetColumnName())));
    }

    private string CreateInsertIntoMemoryTableSql(string insertColumns, List<List<SqlRawProperty>> dataSqlRawProperties)
    {
        var sql = new StringBuilder();
        foreach (var entityRawProperties in dataSqlRawProperties)
        {
            var insertIntoMemoryTableValue =
                CreateInsertIntoMemoryTableValueSql(entityRawProperties, insertColumns);
            sql.AppendLine(insertIntoMemoryTableValue);
        }

        return sql.ToString();
    }

    private static string CreateInsertIntoMemoryTableValueSql(List<SqlRawProperty> rawProperties, string insertColumns)
    {
        var insertValues = CreateInsertValues(rawProperties);
        return $@"INSERT INTO #TempMemoryTable ({insertColumns}) VALUES ({insertValues});";
    }

    private static string CreateInsertValues(List<SqlRawProperty> rawProperties)
    {
        return string.Join(", ", rawProperties.Select(x => $"@p{x.Value.ArgumentIndex}"));
    }

    private string CreateMatchCondition()
    {
        if (_matchExpression == null)
        {
            throw new InvalidOperationException("On Method IsRequired");
        }

        var matchExpressions = GenerateMatchCondition(_matchExpression)
            .Select(x => x.Name)
            .ToList();
        return string.Join("", matchExpressions.Select(x => $"target.{x} = source.{x}"));
    }

    private string CreateMemoryTableSql(string insertColumns, List<List<SqlRawProperty>> dataSqlRawProperties)
    {
        var createMemTableSql = $"CREATE TABLE #TempMemoryTable ({CreateTableColumnsTypes(dataSqlRawProperties[0])});";
        var insertMemTableSql = CreateInsertIntoMemoryTableSql(insertColumns, dataSqlRawProperties);
        return createMemTableSql + "\n" + insertMemTableSql;
    }

    private static string CreateSourceColumns(List<List<SqlRawProperty>> dataSqlRawProperties)
    {
        return string.Join(", ", dataSqlRawProperties[0].Select(x => $"source.[{x.ColumnName}]"));
    }

    private static string CreateTableColumnsTypes(List<SqlRawProperty> rawProperties)
    {
        return string.Join(", ", rawProperties.Select(x => $"[{x.PropertyName}] {x.Value.GetColumnType()}"));
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

    private string GetFullTableName(ISqlGenerationHelper sqlGenerator)
    {
        var tableName = _entityType.GetTableName() ??
                        _entityType.GetDefaultTableName() ?? _entityType.GetType().Name;
        var schema = _entityType.GetSchema();
        var fullTableName = sqlGenerator.DelimitIdentifier(tableName, schema);
        return fullTableName;
    }

    private IEnumerable<SqlRawProperty> GetSqlRawProperties(List<IProperty> properties, TEntity entity)
    {
        return properties.Select((p, index) => p.GetSqlRawProperty(index, entity));
    }
}