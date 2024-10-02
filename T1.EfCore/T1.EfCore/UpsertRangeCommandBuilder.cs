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
        var sqlRawRows = _entityPropertyExtractor.CreateDataSqlRawProperties(properties, _entities)
            .ToList();
        
        var connection = OpenDbConnection();
        ExecuteDbCommand(connection, rowSqlRawProperties.CreateMemTableSql());

        var dataTable = _entityPropertyExtractor.GetSqlColumnProperties(_entityType).CreateDataTable();
        dataTable.AddData(sqlRawRows);
        BulkWriteTable(connection, rowSqlRawProperties, dataTable, "#TempMemoryTable");

        var mergeSql = CreateMergeDataSql(fullTableName, insertColumns, rowSqlRawProperties);
        var sql = mergeSql + "; DROP TABLE #TempMemoryTable;";
        ExecuteDbCommand(connection, sql);
    }

    public UpsertRangeCommandBuilder<TEntity> On(Expression<Func<TEntity, object>> matchExpression)
    {
        _matchExpression = matchExpression;
        return this;
    }

    private static void BulkWriteTable(DbConnection connection, List<SqlRawProperty> rowSqlRawProperties, DataTable dataTable, string targetTable)
    {
        using var bulkCopy = new SqlBulkCopy((SqlConnection)connection, SqlBulkCopyOptions.Default, null);
        bulkCopy.DestinationTableName = targetTable;
        foreach (var column in rowSqlRawProperties)
        {
            bulkCopy.ColumnMappings.Add(column.ColumnName, column.ColumnName);
        }
        bulkCopy.WriteToServer(dataTable);
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

    private static void ExecuteDbCommand(DbConnection connection, string sql)
    {
        using var dbCommand = connection.CreateCommand();
        dbCommand.CommandText = sql;
        dbCommand.ExecuteNonQuery();
    }

    private DbConnection OpenDbConnection()
    {
        var connection = _dbContext.Database.GetDbConnection();
        if(connection.State != ConnectionState.Open)
            connection.Open();
        return connection;
    }
}