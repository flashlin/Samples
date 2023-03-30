using System.Data;
using System.Data.Common;
using System.Text;
using Dapper;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using QueryKits.Entities;
using QueryKits.ExcelUtils;

namespace QueryKits.Services;


public class ReportDbContext : DbContext, IReportRepo
{
    private readonly string _connectionString;

    public ReportDbContext(IOptions<DbConfig> dbConfig)
    {
        _connectionString = dbConfig.Value.ConnectionString;
    }

    public DbSet<SqlHistoryEntity> SqlHistories { get; set; } = null!;

    public List<string> GetAllTableNames()
    {
        var sql = new StringBuilder();
        sql.AppendLine("SELECT TABLE_NAME as TableName");
        sql.AppendLine("FROM INFORMATION_SCHEMA.TABLES");
        sql.Append(@$"WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_CATALOG='{LocalDbService.DatabaseName}'");
        return Database.SqlQueryRaw<string>(sql.ToString()).ToList();
    }

    public List<Dictionary<string, object>> QueryRawSql(string sql)
    {
        using var conn = Database.GetDbConnection();
        return conn.Query(sql)
            .Cast<IDictionary<string, object>>()
            .Select(row => row.ToDictionary(item => item.Key, item => item.Value))
            .ToList();
    }

    public List<QueryDataSet> QueryDapperMultipleRawSql(string sql)
    {
        var conn = Database.GetDbConnection();
        using var multiQuery = conn.QueryMultiple(sql)!;
        var result = new List<QueryDataSet>();
        while (!multiQuery.IsConsumed)
        {
            result.Add(new QueryDataSet
            {
                Rows = multiQuery.Read<Dictionary<string, object>>()
                    .ToList()
            });
        }
        return result;
    }

    public List<QueryDataSet> QueryMultipleRawSql(string sql)
    {
        var conn = Database.GetDbConnection();
        if (conn.State != ConnectionState.Open)
        {
            conn.Open();
        }
        using var command = conn.CreateCommand();
        command.CommandText = sql;
        command.CommandType = CommandType.Text;
        using var reader = command.ExecuteReader();
        var dataSets = new List<QueryDataSet>();
        do
        {
            dataSets.Add(ReadDataSet(reader));
        } while (reader.NextResult());
        return dataSets;
    }

    private static QueryDataSet ReadDataSet(DbDataReader reader)
    {
        var dataSet = new QueryDataSet();
        while (reader.Read())
        {
            var row = new Dictionary<string, object>();
            var unknownCount = 0;
            for (var i = 0; i < reader.FieldCount; i++)
            {
                var columnName = reader.GetName(i);
                if (string.IsNullOrEmpty(columnName))
                {
                    columnName = $"UnknownName{unknownCount}";
                    unknownCount++;
                }
                var columnValue = reader.GetValue(i);
                row[columnName] = columnValue;
            }
            dataSet.Rows.Add(row);
        }
        return dataSet;
    }

    public int ExecuteRawSql(string sql)
    {
        var conn = Database.GetDbConnection();
        return conn.Execute(sql);
    }

    public int DropTable(string tableName)
    {
        var sql = $"IF (OBJECT_ID('{tableName}')) Is Not NULL DROP TABLE [{tableName}];";
        return ExecuteRawSql(sql);
    }

    public void ReCreateTable(string tableName, List<ExcelColumn> headers)
    {
        DropTable(tableName);
        var sql = new StringBuilder();
        sql.Append($"CREATE TABLE");
        sql.Append($"[{tableName}] (");
        sql.Append("_PID INT IDENTITY(1,1),");
        foreach (var header in headers)
        {
            sql.Append($"[{header.Name}] ");
            switch (header.DataType)
            {
                case ExcelDataType.Number:
                    sql.Append("decimal(19,6) NULL");
                    break;
                default:
                    sql.Append("nvarchar(100) NULL");
                    break;
            }

            if (header != headers.Last())
            {
                sql.Append(",");
            }
        }

        sql.Append(")");
        ExecuteRawSql(sql.ToString());
    }

    public int ImportData(string tableName, ExcelSheet sheet)
    {
        var sqlInsertColumns = new StringBuilder();
        sqlInsertColumns.Append($"INSERT [{tableName}](");
        foreach (var header in sheet.Headers)
        {
            sqlInsertColumns.Append($"[{header.Name}]");
            if (header != sheet.Headers.Last())
            {
                sqlInsertColumns.Append(",");
            }
        }
        sqlInsertColumns.Append(")");

        var sql = CreateInsertTableSqlBlock(sqlInsertColumns, sheet.Headers, sheet.Rows);
        return ExecuteRawSql(sql);
    }

    public List<string> GetTop10SqlCode()
    {
        return SqlHistories.OrderByDescending(x => x.CreatedOn)
            .Take(20)
            .Select(x => x.SqlCode)
            .ToList();
    }

    public void AddSqlCode(string sqlCode)
    {
        if (string.IsNullOrEmpty(sqlCode))
        {
            return;
        }
        SqlHistories.Add(new SqlHistoryEntity
        {
            SqlCode = sqlCode,
            CreatedOn = DateTime.Now
        });
        SaveChanges();
    }

    private static string CreateInsertTableSqlBlock(StringBuilder sqlInsertColumns, List<ExcelColumn> headers, List<Dictionary<string, string>> rows)
    {
        var sqlBlock = new StringBuilder();
        foreach (var row in rows)
        {
            var sqlInsert = sqlInsertColumns.ToString() + " " + CreateValuesSql(headers, row) + "\r\n";
            sqlBlock.Append(sqlInsert);
        }

        return sqlBlock.ToString();
    }

    private static string CreateValuesSql(List<ExcelColumn> headers, Dictionary<string, string> row)
    {
        var sql = new StringBuilder();
        sql.Append("VALUES (");
        foreach (var header in headers)
        {
            var value = row[header.Name];
            if (header.DataType == ExcelDataType.String)
            {
                value = "N'" + value + "'";
            }
            sql.Append(value);

            if (header != headers.Last())
            {
                sql.Append(",");
            }
        }

        sql.Append(")");
        return sql.ToString();
    }

    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        optionsBuilder.UseSqlServer(_connectionString);
    }
}