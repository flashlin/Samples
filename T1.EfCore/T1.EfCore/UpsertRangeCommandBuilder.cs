using System.Data;
using System.Data.Common;
using System.Linq.Expressions;
using Microsoft.Data.SqlClient;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Metadata;
using Microsoft.EntityFrameworkCore.Storage;

namespace T1.EfCore;

public class UpsertRangeCommandBuilder<TEntity> where TEntity : class
{
    private readonly BulkInsertCommandBuilder<TEntity> _bulkInsertCommandBuilder;
    private readonly DbContext _dbContext;
    private readonly List<TEntity> _entities;
    private readonly EntityPropertyExtractor _entityPropertyExtractor = new();
    private readonly IEntityType _entityType;
    private readonly EntityTypeMatchConditionGenerator<TEntity> _entityTypeMatchConditionGenerator = new();
    private Expression<Func<TEntity, object>>? _matchExpression;

    public UpsertRangeCommandBuilder(DbContext dbContext, IEntityType entityType, IEnumerable<TEntity> entities)
    {
        _dbContext = dbContext;
        _entityType = entityType;
        _entities = entities.ToList();
        _bulkInsertCommandBuilder = new BulkInsertCommandBuilder<TEntity>(dbContext, _entities);
    }

    public void Execute()
    {
        var sqlGenerator = _dbContext.GetService<ISqlGenerationHelper>();
        var fullTableName = sqlGenerator.GetFullTableName(_entityType);

        var properties = _entityType.GetProperties().ToList();
        var insertColumns = CreateInsertColumns(sqlGenerator, properties);
        var rowSqlRawProperties = _entityPropertyExtractor.GetSqlRawProperties(properties, _entities[0])
            .ToList();
        
        var dataSqlRawProperties = _entityPropertyExtractor.CreateDataSqlRawProperties(properties, _entities)
            .ToList();
        var dataTable = _entityPropertyExtractor.GetSqlColumnProperties(_entityType).CreateDataTable();
        dataTable.AddData(dataSqlRawProperties);
        
        var connection = OpenDbConnection();

        using var dbCommand = connection.CreateCommand();
        var createTempMemTableSql = rowSqlRawProperties.CreateMemTableSql();
        dbCommand.CommandText = createTempMemTableSql;
        dbCommand.ExecuteNonQuery();
        
        using var bulkCopy = new SqlBulkCopy((SqlConnection)connection, SqlBulkCopyOptions.Default, null);
        bulkCopy.DestinationTableName = "#TempMemoryTable";
        foreach (var column in rowSqlRawProperties)
        {
            bulkCopy.ColumnMappings.Add(column.ColumnName, column.ColumnName);
        }
        bulkCopy.WriteToServer(dataTable);
        
        using var dbCommand2 = connection.CreateCommand();
        var mergeSql = CreateMergeDataSql(fullTableName, insertColumns, rowSqlRawProperties);
        dbCommand2.CommandText = mergeSql + "; DROP TABLE #TempMemoryTable;";
        dbCommand2.ExecuteNonQuery();
    }

    public UpsertRangeCommandBuilder<TEntity> On(Expression<Func<TEntity, object>> matchExpression)
    {
        _matchExpression = matchExpression;
        return this;
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
        var matchExpressions = _entityTypeMatchConditionGenerator.GenerateMatchCondition(_entityType, _matchExpression)
            .Select(x => x.Name)
            .ToList();
        return string.Join(" and ", matchExpressions.Select(x => $"target.{x} = source.{x}"));
    }

    private string CreateMergeDataSql(string fullTableName, string insertColumns, List<SqlRawProperty> row)
    {
        var sourceColumns = row.CreateSourceColumns();
        var matchCondition = CreateMatchCondition();
        var mergeSql = $@"MERGE INTO {fullTableName} AS target
USING #TempMemoryTable AS source
ON ({matchCondition})
WHEN NOT MATCHED THEN
    INSERT ({insertColumns})
    VALUES ({sourceColumns});";
        return mergeSql;
    }

    private DbConnection OpenDbConnection()
    {
        var connection = _dbContext.Database.GetDbConnection();
        if(connection.State != ConnectionState.Open)
            connection.Open();
        return connection;
    }
}