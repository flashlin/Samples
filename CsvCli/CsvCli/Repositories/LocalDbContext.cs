using System.Data;
using System.Globalization;
using System.Text;
using CsvHelper;
using CsvHelper.Configuration;
using Dapper;
using Microsoft.EntityFrameworkCore;
using T1.Standard.Data;
using T1.Standard.Extensions;

namespace CsvCli.Repositories;

public class LocalDbContext : DbContext
{
    private readonly string _sqliteFile = "local.db";

    /*
    public DbSet<StockEntity> StocksMap { get; set; }
    public DbSet<TransEntity> Trans { get; set; }
    public DbSet<StockHistoryEntity> StocksHistory { get; set; }
    */

    public void ImportCsvFile(string csvFile, string tableName)
    {
        var readConfiguration = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = true
        };
        using var reader = new StreamReader(csvFile, Encoding.UTF8);
        using var csv = new CsvReader(reader, readConfiguration);
        using var dr = new CsvDataReader(csv);
        var dataTable = new DataTable();
        dataTable.Load(dr);
        dataTable.TableName = tableName;

        if (!IsTableExists(tableName))
        {
            CreateTable(dataTable, tableName);
        }

        var dictObjList = dataTable.ToDictionary();

        var insertSqlCode = GenerateInsertSqlCode(dataTable);
        using var connection = Database.GetDbConnection();
        connection.Open();
        using var transaction = connection.BeginTransaction();
        foreach (var dictObj in dictObjList)
        {
            //var dyObj = dictObj.ConvertToObject<dynamic>();
            connection.Execute(insertSqlCode, dictObj);
        }
        transaction.Commit();
    }

    private string GenerateInsertSqlCode(DataTable dataTable)
    {
        var code = new StringBuilder();
        code.Append($"INSERT INTO {dataTable.TableName}");
        code.Append("(");
        code.Append(string.Join(", ", GetDataColumnsType(dataTable).Select(x => x.Name)));
        code.AppendLine(")");
        code.Append("VALUES (");
        code.Append(string.Join(", ", GetDataColumnsType(dataTable).Select(x => $"@{x.Name}")));
        code.Append(")");
        return code.ToString();
    }

    private void CreateTable(DataTable dt, string tableName)
    {
        var code = GenerateCreateTableSqlCode(tableName, dt);
        ExecuteRaw(code, null);
    }

    public bool IsTableExists(string tableName)
    {
        var sql = @"SELECT 1 FROM sqlite_master WHERE type='table' AND name=@tableName";
        var result = QueryRaw<long>(sql, new {tableName}).FirstOrDefault();
        return result != 0;
    }

    /*protected T QueryScalar<T>(string sql, object queryParameter)
    {
        var connection = Database.GetDbConnection();
        var q1 = connection.Query(sql, queryParameter);
        var dapperList = q1.FirstOrDefault();
        var dictList = dapperList.Select(x => (IDictionary<string, object>) x);
        foreach (var dict in dictList)
        {
            var item = dict.ConvertToObject<T>();
            yield return item;
        }
    }*/

    protected IEnumerable<T> QueryRaw<T>(string sql, object? queryParameter = null)
        where T : new()
    {
        var connection = Database.GetDbConnection();
        var q1 = connection.Query(sql, queryParameter);

        var dapperList = q1.ToList();
        var dictList = dapperList.Select(x => (IDictionary<string, object>) x);
        if (!typeof(T).IsClass)
        {
            return dictList.Select(x => (T) x.First().Value);
        }

        return dictList.Select(x => x.ConvertToObject<T>());
    }

    private IEnumerable<DataColumnType> GetDataColumnsType(DataTable dataTable)
    {
        for (var i = 0; i < dataTable.Columns.Count; i++)
        {
            var column = dataTable.Columns[i];
            var name = column.ColumnName;
            var dataType = GetDataType(column.DataType);
            yield return new DataColumnType
            {
                Name = name,
                DataType = dataType
            };
        }
    }

    private string GenerateCreateTableSqlCode(string tableName, DataTable dataTable)
    {
        var code = new StringBuilder();
        code.AppendLine($"CREATE TABLE {tableName} (");
        var columnDefines = string.Join(",\r\n", GetDataColumnsType(dataTable).Select(x => $"{x.Name} {x.DataType}"));
        code.AppendLine(columnDefines);
        code.AppendLine(")");
        return code.ToString();
    }

    protected void ExecuteRaw(string sql, object? queryParameter)
    {
        var connection = Database.GetDbConnection();
        connection.Execute(sql, queryParameter);
    }

    private string GetDataType(Type columnDataType)
    {
        if (columnDataType == typeof(string))
        {
            return "VARCHAR(50)";
        }

        if (columnDataType == typeof(DateTime))
        {
            return "DATETIME";
        }

        if (columnDataType == typeof(int))
        {
            return "INT";
        }

        return "DECIMAL(10,2)";
    }

    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        optionsBuilder.UseSqlite($"DataSource={_sqliteFile};")
            .UseQueryTrackingBehavior(QueryTrackingBehavior.NoTracking);
    }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        /*
        modelBuilder.Entity<StockHistoryEntity>(entity =>
        {
            entity.HasKey(e => new { e.TranDate, e.StockId  });
        });
        modelBuilder.Entity<TransEntity>()
            .Property(e => e.Balance)
            .HasConversion<double>();
    */
    }
}

public class DataColumnType
{
    public string Name { get; set; }
    public string DataType { get; set; }
}