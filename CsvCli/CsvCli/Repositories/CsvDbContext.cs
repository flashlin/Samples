using System.Data;
using System.Globalization;
using System.Text;
using CsvCli.Helpers;
using CsvHelper;
using CsvHelper.Configuration;
using Dapper;
using Microsoft.EntityFrameworkCore;
using T1.Standard.Common;
using T1.Standard.Data;
using T1.Standard.Extensions;

namespace CsvCli.Repositories;

public class CsvDbContext : DbContext
{
    private readonly string _sqliteFile = "local.db";

    /*
    public DbSet<StockEntity> StocksMap { get; set; }
    public DbSet<TransEntity> Trans { get; set; }
    public DbSet<StockHistoryEntity> StocksHistory { get; set; }
    */

    public void ImportCsvFile(string csvFile, string tableName)
    {
        var dataTable = ReadCsvFileToDataTable(csvFile, tableName);
        dataTable = AdjustDataTable(dataTable);

        if (!IsTableExists(tableName))
        {
            Console.WriteLine($"Create Table {tableName}");
            CreateTable(dataTable, tableName);
        }

        var dictObjList = dataTable.ToDictionary();
        var insertSqlCode = GenerateInsertSqlCode(dataTable);
        BulkExecute(dictObjList, insertSqlCode);
    }

    public void QuerySqlCode(string sqlCode)
    {
        var connection = Database.GetDbConnection();
        var q1 = connection.Query(sqlCode)
            .Select(x => (IDictionary<string, object>)x)
            .ToArray();

        var maxLength = GetMaxLength(q1);
        var first = true;
        foreach (var dict in q1)
        {
            if (first)
            {
                var titleLine = string.Join(" ",
                    dict.Keys.Select(x => x.ToBig5FixLenString(maxLength[x])));
                Console.WriteLine(titleLine);
                first = false;
            }

            var line = string.Join(" ",
                dict.Select(x => $"{x.Value}".ToBig5FixLenString(maxLength[x.Key])));
            Console.WriteLine(line);
        }
    }

    public string GetStringFixed(string text, int maxLen)
    {
        var len = maxLen - text.Length;
        if (len > 0)
        {
            var spaces = new string(' ', len);
            return text + spaces;
        }

        return text;
    }

    private static Dictionary<string, int> GetMaxLength(IEnumerable<IDictionary<string, object>> q1)
    {
        var maxLenDict = new Dictionary<string, int>();
        var first = true;
        foreach (IDictionary<string, object> row in q1)
        {
            if (first)
            {
                foreach (var key in row.Keys)
                {
                    maxLenDict[key] = key.Length;
                }

                first = false;
            }

            foreach (var key in row.Keys)
            {
                maxLenDict[key] = Math.Max(maxLenDict[key], $"{row[key]}".Length);
            }
        }

        return maxLenDict;
    }

    public static DataTable AdjustDataTable(DataTable dataTable)
    {
        var newColumnList = GetDataColumnsAdjusted(dataTable).ToArray();
        var newDataTable = new DataTable(dataTable.TableName);
        foreach (var column in newColumnList)
        {
            newDataTable.Columns.Add(column);
        }

        Console.WriteLine($"AdjustDataTable {dataTable.Rows.Count}");
        foreach (DataRow row in dataTable.Rows)
        {
            var newRow = newDataTable.NewRow();
            foreach (var newColumn in newColumnList)
            {
                var value = $"{row[newColumn.ColumnName]}";
                if (newColumn.DataType == typeof(DateTime))
                {
                    newRow[newColumn] = value.ChangeType(newColumn.DataType);
                    continue;
                }

                if (newColumn.DataType == typeof(decimal))
                {
                    value = value.Replace(",", "");
                    if (string.IsNullOrEmpty(value))
                    {
                        newRow[newColumn] = 0m;
                    }
                    else
                    {
                        newRow[newColumn] = value.ChangeType(newColumn.DataType);
                    }

                    continue;
                }

                if (newColumn.DataType != typeof(string))
                {
                    newRow[newColumn] = value.ChangeType(newColumn.DataType);
                    continue;
                }

                newRow[newColumn] = value;
            }

            newDataTable.Rows.Add(newRow);
        }

        return newDataTable;
    }

    public static IEnumerable<DataColumn> GetDataColumnsAdjusted(DataTable dataTable)
    {
        var row = dataTable.Rows[0];
        foreach (DataColumn column in dataTable.Columns)
        {
            var value = $"{row[column]}";
            if (DateTime.TryParse(value, out _))
            {
                yield return new DataColumn(column.ColumnName, typeof(DateTime));
                continue;
            }

            if (value.Contains(',') && decimal.TryParse(value.Replace(",", ""), out _))
            {
                yield return new DataColumn(column.ColumnName, typeof(decimal));
                continue;
            }

            if (long.TryParse(value, out _))
            {
                yield return new DataColumn(column.ColumnName, typeof(long));
                continue;
            }

            yield return new DataColumn(column.ColumnName, typeof(string));
        }
    }

    private static DataTable ReadCsvFileToDataTable(string csvFile, string tableName)
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

        foreach (DataColumn column in dataTable.Columns)
        {
            column.ColumnName = column.ColumnName.Replace(" ", "");
            column.ColumnName = column.ColumnName.Replace("(", "");
            column.ColumnName = column.ColumnName.Replace(")", "");
        }
        
        
        Console.WriteLine("Load CSV to DataTable");
        return dataTable;
    }

    private void BulkExecute(IEnumerable<Dictionary<string, object>> dictObjList, string insertSqlCode)
    {
        using var connection = Database.GetDbConnection();
        connection.Open();
        using var transaction = connection.BeginTransaction();
        foreach (var dictObj in dictObjList)
        {
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
        var result = QueryRaw<long>(sql, new { tableName }).FirstOrDefault();
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

    private IEnumerable<T> QueryRaw<T>(string sql, object? queryParameter = null)
        where T : new()
    {
        var connection = Database.GetDbConnection();
        var q1 = connection.Query(sql, queryParameter);

        var dapperList = q1.ToList();
        var dictList = dapperList.Select(x => (IDictionary<string, object>)x);
        if (!typeof(T).IsClass)
        {
            return dictList.Select(x => (T)x.First().Value);
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

        if (columnDataType == typeof(long))
        {
            return "LONG";
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