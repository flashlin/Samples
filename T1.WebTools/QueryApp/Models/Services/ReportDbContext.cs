using System.Text;
using Dapper;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Primitives;
using QueryApp.Models.Helpers;

namespace QueryApp.Models.Services;

public class ReportDbContext : DbContext, IReportRepo
{
    private readonly string _connectionString;

    public ReportDbContext(ILocalDbService localDbService)
    {
        _connectionString = localDbService.GetDbConnectionString();
    }

    public List<string> GetAllTableNames()
    {
        var sql = $"""
            SELECT TABLE_NAME as TableName 
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_CATALOG='{LocalDbService.DatabaseName}'
            """;
        return Database.SqlQueryRaw<string>(sql)
            .ToList();
    }

    public List<Dictionary<string, object>> QueryRawSql(string sql)
    {
        using var conn = Database.GetDbConnection();
        return conn.Query(sql)
            .Cast<IDictionary<string, object>>()
            .Select(row => row.ToDictionary(item => item.Key, item => item.Value))
            .ToList();
    }

    public int DropTable(string tableName)
    {
        var sql = $"IF (OBJECT_ID('{tableName}')) Is Not NULL DROP TABLE [{tableName}];";
        return SqlQueryRaw(sql).First();
    }

    public void ReCreateTable(string tableName, List<ExcelColumn> headers)
    {
        DropTable(tableName);
        var sql = new StringBuilder();
        sql.Append($"CREATE TABLE [{tableName}] (");
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
        Database.SqlQueryRaw<int>(sql.ToString());
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
        return Database.SqlQueryRaw<int>(sql).First();
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
                value = "'" + value + "'";
            }

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