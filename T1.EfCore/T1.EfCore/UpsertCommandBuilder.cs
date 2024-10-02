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
    private readonly EntityPropertyExtractor _entityPropertyExtractor = new (); 
    private readonly SqlBuilder _sqlBuilder = new (); 

    public UpsertCommandBuilder(DbContext dbContext, IEntityType entityType, params TEntity[] entities)
    {
        _dbContext = dbContext;
        _entityType = entityType;
        _entityList = entities;
    }

    public void Execute()
    {
        var sqlGenerator = _dbContext.GetService<ISqlGenerationHelper>();
        var fullTableName = sqlGenerator.GetFullTableName(_entityType);

        var properties = _entityType.GetProperties().ToList();
        var dataSqlRawProperties = _entityPropertyExtractor.CreateDataSqlRawProperties(properties, _entityList)
            .ToList();
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

    private List<DbParameter> CreateDbParameters(DbCommand dbCommand, List<SqlRawProperty> entitySqlRawProperties)
    {
        return entitySqlRawProperties.Select(x =>
        {
            var dbCommandArgumentBuilder = new DbCommandArgumentBuilder(_dbContext, dbCommand);
            return dbCommandArgumentBuilder.CreateDbParameter(x.DataValue);
        }).ToList();
    }

    private static string CreateInsertColumns(ISqlGenerationHelper sqlGenerator, List<IProperty> properties)
    {
        return string.Join(", ", properties.Select(p => sqlGenerator.DelimitIdentifier(p.GetColumnName())));
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
        var createMemTempTableSql = dataSqlRawProperties.CreateAndInsertMemTempTableSql(insertColumns);
        var sourceColumns = _sqlBuilder.CreateColumns("source", dataSqlRawProperties[0]);
        var matchCondition = CreateMatchCondition();
        var mergeSql = $@"{createMemTempTableSql}
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
        var insertValues = string.Join(", ", dataSqlRawProperties.Select(x=> $"@p{x.DataValue.ArgumentIndex}"));
        var mergeSql = $@"
MERGE INTO {fullTableName} AS target
USING (SELECT {insertValues}) AS source({insertColumns}) 
ON ({matchCondition})
WHEN NOT MATCHED THEN
    INSERT ({insertColumns})
    VALUES ({insertValues});";
        return mergeSql;
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
}