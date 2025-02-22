using System.Data.Common;
using System.Linq.Expressions;
using System.Text;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Metadata;
using Microsoft.EntityFrameworkCore.Storage;

namespace T1.EfCore;

public class UpsertCommandBuilder<TEntity> where TEntity : class
{
    private readonly DbContext _dbContext;
    private readonly TEntity[] _entityArray;
    private readonly IEntityType _entityType;
    private readonly SqlBuilder _sqlBuilder = new ();
    private readonly SqlRawPropertyBuilder _sqlRawPropertyBuilder = new ();
    private Expression<Func<TEntity, object>>? _matchExpression;

    public UpsertCommandBuilder(DbContext dbContext, IEntityType entityType, params TEntity[] entities)
    {
        _dbContext = dbContext;
        _entityType = entityType;
        _entityArray = entities;
    }

    public int Execute()
    {
        var sqlGenerator = _dbContext.GetService<ISqlGenerationHelper>();

        var rowProperties = _entityType.GetProperties().ToList();
        var sqlRawData = _sqlRawPropertyBuilder.CreateSqlRawData(rowProperties, _entityArray)
            .ToList();
        var insertColumns = sqlGenerator.CreateInsertColumnsSql(rowProperties);
        var fullTableName = sqlGenerator.GetFullTableName(_entityType);
        var mergeSql = CreateMergeDataSql(fullTableName, insertColumns, sqlRawData);

        using var dbCommand = _dbContext.Database.GetDbConnection().CreateCommand();
        var values = CreateDataDbParameters(_dbContext, dbCommand, sqlRawData)
            .ToList();
        dbCommand.Connection?.Open();
        dbCommand.CommandText = mergeSql;
        dbCommand.Parameters.AddRange(values.ToArray());
        using var reader = dbCommand.ExecuteReader();
        var effectedRowCount = 0;
        if (reader.Read())
        {
            effectedRowCount = reader.GetInt32(0);
        }
        return effectedRowCount;
        //return _dbContext.Database.ExecuteSqlRaw(mergeSql, values);
    }

    public UpsertCommandBuilder<TEntity> On(Expression<Func<TEntity, object>> matchExpression)
    {
        _matchExpression = matchExpression;
        return this;
    }

    private IEnumerable<DbParameter> CreateDataDbParameters(DbContext dbContext, DbCommand dbCommand,
        List<List<SqlRawProperty>> dataSqlRawProperties)
    {
        foreach (var entitySqlRawProperties in dataSqlRawProperties)
        {
            var dbParameters = CreateDbParameters(dbContext, dbCommand, entitySqlRawProperties);
            foreach (var dbParameter in dbParameters)
            {
                yield return dbParameter;
            }
        }
    }

    private List<DbParameter> CreateDbParameters(DbContext dbContext, DbCommand dbCommand,
        List<SqlRawProperty> entitySqlRawProperties)
    {
        return entitySqlRawProperties.Select(x =>
        {
            var dbCommandArgumentBuilder = new DbCommandArgumentBuilder(dbContext, dbCommand);
            return dbCommandArgumentBuilder.CreateDbParameter(x.DataValue);
        }).ToList();
    }

    private string CreateMatchConditionSql()
    {
        if (_matchExpression == null)
        {
            throw new InvalidOperationException("On Method IsRequired");
        }
        return _sqlBuilder.CreateMatchConditionSql(_entityType, _matchExpression);
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
        var matchCondition = CreateMatchConditionSql();
        var mergeSql = $@"{createMemTempTableSql}
MERGE INTO {fullTableName} AS target
USING #TempMemoryTable AS source
ON ({matchCondition})
WHEN NOT MATCHED THEN
    INSERT ({insertColumns})
    VALUES ({sourceColumns});
SELECT @@ROWCOUNT AS InsertedRows;";
        return mergeSql;
    }

    private string CreateMergeSingleDataSql(string fullTableName, string insertColumns,
        List<SqlRawProperty> rowProperties)
    {
        var matchCondition = CreateMatchConditionSql();
        var insertValues = string.Join(", ", rowProperties.Select(x=> $"@p{x.DataValue.ArgumentIndex}"));
        var mergeSql = $@"
MERGE INTO {fullTableName} AS target
USING (SELECT {insertValues}) AS source({insertColumns}) 
ON ({matchCondition})
WHEN NOT MATCHED THEN
    INSERT ({insertColumns})
    VALUES ({insertValues});
SELECT @@ROWCOUNT AS TotalAffectedRows;";
        return mergeSql;
    }
}