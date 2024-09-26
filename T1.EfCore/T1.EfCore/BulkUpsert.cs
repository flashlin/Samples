using System.Data;
using System.Linq.Expressions;
using Microsoft.EntityFrameworkCore;
using Microsoft.Data.SqlClient;
using Microsoft.EntityFrameworkCore.Metadata;

namespace T1.EfCore;

public class BulkInserter<TEntity>
    where TEntity : class
{
    private readonly DbContext _dbContext;
    private readonly EntityPropertyExtractor _entityPropertyExtractor = new ();
    private readonly IEntityType _entityType;
    private List<SqlColumnProperty> _properties;

    public BulkInserter(DbContext dbContext, IEntityType entityType)
    {
        _dbContext = dbContext;
        _entityType = entityType;
        Initialize();
    }

    public void BulkInsert(List<TEntity> entities, string tableName)
    {
        var properties = _properties.Select(x => x.Property).ToList();
        var dataSqlRawProperties = _entityPropertyExtractor.CreateDataSqlRawProperties(properties, entities)
            .ToList();

        var dataTable = new DataTable();
        foreach (var column in _properties)
        {
            var dataColumn = new DataColumn(column.ColumnName, column.Property.ClrType); 
            dataTable.Columns.Add(dataColumn);
        }
        foreach (var entity in dataSqlRawProperties)
        {
            var row = dataTable.NewRow();
            foreach (var prop in entity)
            {
                row[prop.ColumnName] = prop.DataValue.Value;
            }
            dataTable.Rows.Add(row);
        }
        
        var connection = _dbContext.Database.GetDbConnection();
        using var bulkCopy = new SqlBulkCopy((SqlConnection)connection, SqlBulkCopyOptions.Default, null);
        bulkCopy.DestinationTableName = tableName;
        foreach (var column in _properties)
        {
            bulkCopy.ColumnMappings.Add(column.ColumnName, column.ColumnName);
        }
        bulkCopy.WriteToServer(dataTable);
    }

    private void Initialize()
    {
        _properties = _entityType.GetProperties().Select(x =>
            new SqlColumnProperty()
            {
                Property = x,
                ColumnName = x.GetColumnName(),
                AllowInsert = x.IsAllowInsert()
            }).ToList();
    }
}