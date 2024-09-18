using System.Data.Common;
using System.Linq.Expressions;
using System.Reflection;
using System.Text;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Metadata;
using Microsoft.EntityFrameworkCore.Storage;

namespace T1.EfCore;

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
        
        var mergeSql = CreateMergeDataSql(fullTableName, insertColumns, dataSqlRawProperties);

        using var dbCommand = _dbContext.Database.GetDbConnection().CreateCommand();
        var values = CreateDataDbParameters(dbCommand, dataSqlRawProperties)
            .ToList();
        _dbContext.Database.ExecuteSqlRaw(mergeSql, values);
    }

    public UpsertCommandBuilder<TEntity> On(Expression<Func<TEntity, object>> matchExpression)
    {
        _matchExpression = matchExpression;
        return this;
    }

    private IEnumerable<DbParameter> CreateDataDbParameters(DbCommand dbCommand, List<List<SqlRawProperty>> dataSqlRawProperties)
    {
        foreach (var entitySqlRawProperties in dataSqlRawProperties)
        {
            var dbParameters = CreateDbParameters(dbCommand, entitySqlRawProperties);
            foreach (var dbParameter in dbParameters)
            {
                yield return dbParameter;
            }
        }
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

    private List<DbParameter> CreateDbParameters(DbCommand dbCommand, List<SqlRawProperty> entitySqlRawProperties)
    {
        return entitySqlRawProperties.Select(x =>
        {
            var dbCommandArgumentBuilder = new DbCommandArgumentBuilder(_dbContext, dbCommand);
            return dbCommandArgumentBuilder.CreateDbParameter(x.Value);
        }).ToList();
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
        return string.Join(" and ", matchExpressions.Select(x => $"target.{x} = source.{x}"));
    }

    private string CreateMemoryTableSql(string insertColumns, List<List<SqlRawProperty>> dataSqlRawProperties)
    {
        var createMemTableSql = $"CREATE TABLE #TempMemoryTable ({CreateTableColumnsTypes(dataSqlRawProperties[0])});";
        var insertMemTableSql = CreateInsertIntoMemoryTableSql(insertColumns, dataSqlRawProperties);
        return createMemTableSql + "\n" + insertMemTableSql;
    }

    private string CreateMergeDataSql(string fullTableName, string insertColumns, List<List<SqlRawProperty>> dataSqlRawProperties)
    {
        if (dataSqlRawProperties.Count == 1)
        {
            return CreateMergeSingleDataSql(fullTableName, insertColumns, dataSqlRawProperties[0]);
        }
        return CreateMergeMultipleDataSql(fullTableName, insertColumns, dataSqlRawProperties);
    }

    private string CreateMergeMultipleDataSql(string fullTableName, string insertColumns,
        List<List<SqlRawProperty>> dataSqlRawProperties)
    {
        var createMemoryTableSql = CreateMemoryTableSql(insertColumns, dataSqlRawProperties);
        var sourceColumns = CreateSourceColumns(dataSqlRawProperties);
        var matchCondition = CreateMatchCondition();
        var mergeSql = $@"{createMemoryTableSql}
MERGE INTO {fullTableName} AS target
USING #TempMemoryTable AS source
ON ({matchCondition})
WHEN NOT MATCHED THEN
    INSERT ({insertColumns})
    VALUES ({sourceColumns});";
        return mergeSql;
    }

    private string CreateMergeSingleDataSql(string fullTableName, string insertColumns,
        List<SqlRawProperty> dataSqlRawProperties)
    {
        var matchCondition = CreateMatchCondition();
        var insertValues = string.Join(", ", dataSqlRawProperties.Select(x=> $"@p{x.Value.ArgumentIndex}"));
        var mergeSql = $@"
MERGE INTO {fullTableName} AS target
USING (SELECT {insertValues}) AS source({insertColumns}) 
ON ({matchCondition})
WHEN NOT MATCHED THEN
    INSERT ({insertColumns})
    VALUES ({insertValues});";
        return mergeSql;
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
        
        if (matchExpression.Body is NewExpression newExpression)
        {
            var joinColumns = new List<IProperty>();
            foreach (var expression in newExpression.Arguments)
            {
                var arg = (MemberExpression)expression;
                if (arg is not { Member: PropertyInfo } || typeof(TEntity) != arg.Expression?.Type)
                    throw new InvalidOperationException("MatchColumns Have To Be Properties Of The EntityClass");
                var property = _entityType.FindProperty(arg.Member.Name);
                if (property == null)
                    throw new InvalidOperationException($"UnknownProperty {arg.Member.Name}");
                joinColumns.Add(property);
            }
            return joinColumns;
        }

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