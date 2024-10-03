using System.Data;
using System.Linq.Expressions;
using Microsoft.EntityFrameworkCore;
using Microsoft.Data.SqlClient;
using Microsoft.EntityFrameworkCore.Infrastructure;
using Microsoft.EntityFrameworkCore.Metadata;
using Microsoft.EntityFrameworkCore.Storage;

namespace T1.EfCore;

public class BulkInsertCommandBuilder<TEntity>
    where TEntity : class
{
    private readonly DbContext _dbContext;
    private readonly IEnumerable<TEntity> _entities;
    private readonly SqlRawPropertyExtractor _sqlRawPropertyExtractor = new ();
    private IEntityType? _entityType;
    private List<SqlColumnProperty> _properties = [];
    private string _tableName = string.Empty;

    public BulkInsertCommandBuilder(DbContext dbContext, IEnumerable<TEntity> entities)
    {
        _entities = entities;
        _dbContext = dbContext;
    }

    public void Execute()
    {
        var entityList = _entities.ToList();
        if (string.IsNullOrEmpty(_tableName))
        {
            var sqlGenerator = _dbContext.GetService<ISqlGenerationHelper>();
            var entityType = ExtractEntityType(entityList[0]);
            _tableName = sqlGenerator.GetFullTableName(entityType);
        }
        var rowProperties = _properties.Select(x => x.Property).ToList();
        var dataSqlRawProperties = _sqlRawPropertyExtractor.CreateSqlRawData(rowProperties, entityList)
            .ToList();

        var dataTable = _properties.CreateDataTable();
        dataTable.AddData(dataSqlRawProperties);
        
        var connection = _dbContext.Database.GetDbConnection();
        if(connection.State != ConnectionState.Open)
            connection.Open();
        using var bulkCopy = new SqlBulkCopy((SqlConnection)connection, SqlBulkCopyOptions.Default, null);
        bulkCopy.DestinationTableName = _tableName;
        foreach (var column in _properties)
        {
            bulkCopy.ColumnMappings.Add(column.ColumnName, column.ColumnName);
        }
        bulkCopy.WriteToServer(dataTable);
    }

    public BulkInsertCommandBuilder<TEntity> Into(string tableName)
    {
        _tableName = tableName;
        return this;
    }

    private IEntityType ExtractEntityType(TEntity entity)
    {
        if (_entityType != null)
        {
            return _entityType;
        }
        _entityType = _dbContext.GetEntityType(entity);
        _properties = _entityType.GetProperties().Select(x =>
            new SqlColumnProperty()
            {
                Property = x,
                ColumnName = x.GetColumnName(),
                AllowInsert = x.IsAllowInsert()
            }).ToList();
        return _entityType;
    }
}