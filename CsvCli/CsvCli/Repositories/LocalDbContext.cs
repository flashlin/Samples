using System.Data;
using System.Globalization;
using System.Text;
using CsvHelper;
using CsvHelper.Configuration;
using Dapper;
using Microsoft.EntityFrameworkCore;
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
        var dt = new DataTable();
        dt.Load(dr);

        if (!IsTableExists(tableName))
        {
            CreateTable(dt, tableName);
        }
    }

    private void CreateTable(DataTable dt, string tableName)
    {
        var code = GenerateCreateTableSqlCode(tableName, dt);
        ExecuteRaw(code, null);
    }

    public bool IsTableExists(string tableName)
    {
        var sql = @"SELECT 1 FROM sqlite_master WHERE type='table' AND name=@tableName";
        return QueryRaw<long>(sql, new { tableName }).First() == 1 ? true : false;
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

    protected IEnumerable<T> QueryRaw<T>(string sql, object? queryParameter=null)
        where T: new()
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

    private string GenerateCreateTableSqlCode(string tableName, DataTable dataTable)
    {
        var code = new StringBuilder();
        code.AppendLine($"CREATE TABLE {tableName} (");
        for (var i = 0; i < dataTable.Columns.Count; i++)
        {
            if (i != 0)
            {
                code.AppendLine(",");
            }

            var column = dataTable.Columns[i];
            code.Append(GetColumnDefine(column));
        }

        code.AppendLine(")");
        return code.ToString();
    }


    protected void ExecuteRaw(string sql, object? queryParameter)
    {
        var connection = Database.GetDbConnection();
        connection.Execute(sql, queryParameter);
    }

    private string GetColumnDefine(DataColumn column)
    {
        var name = column.ColumnName;
        var dataType = GetDataType(column.DataType);
        return $"{name} {dataType}";
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
        optionsBuilder.UseSqlite($"DataSource={_sqliteFile};");
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