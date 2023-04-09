using System.Data;
using System.Data.Common;
using System.Text;
using Dapper;
using Microsoft.Data.SqlClient;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Primitives;
using QueryKits.Entities;
using QueryKits.ExcelUtils;
using T1.Standard.Collections.Generics;
using T1.Standard.Data.SqlBuilders;

namespace QueryKits.Services;

public class ReportDbContext : DbContext, IReportRepo
{
    private const string DatabaseName = "QueryDb";

    public ReportDbContext(IDbContextOptionsFactory factory)
        : base(factory.Create<ReportDbContext>())
    {
        SqlBuilder = factory.CreateSqlBuilder();
    }

    public ISqlBuilder SqlBuilder { get; }

    public DbSet<SqlHistoryEntity> SqlHistories { get; set; } = null!;

    public void CreateTableByEntity(Type entityType)
    {
        var sql = SqlBuilder.CreateTableStatement(SqlBuilder.GetTableInfo(entityType));
        Execute(sql);
    }

    public void MergeTable(MergeTableRequest req)
    {
        var sql = SqlBuilder.CreateMergeTableStatement(req);
        Execute(sql);
    }

    public List<string> GetAllTableNames()
    {
        var sql = SqlBuilder.GetAllTableNamesStatement(DatabaseName);
        return Query<string>(sql)
            .ToList();
    }

    public TableInfo GetTableInfo(string tableName)
    {
        var sql = SqlBuilder.GetTableInfoStatement(tableName);
        var columns = Query<TableColumnInfoEntity>(sql).ToList();
        return new TableInfo
        {
            Name = tableName,
            Columns = columns.Select(x => new TableColumnInfo
            {
                IsKey = x.IsKey,
                IsAutoIncrement = x.IsIdentity,
                Name = x.Name,
                DataType = new DataTypeInfo
                {
                    TypeName = x.DataType,
                    Precision = x.Precision ?? x.Size!.Value,
                    Scale = x.Scale ?? 0
                }
            }).ToList()
        };
    }

    public List<Dictionary<string, object>> QueryRawSql(string sql)
    {
        var conn = GetDbConnection();
        return conn.Query(sql)
            .Cast<IDictionary<string, object>>()
            .Select(row => row.ToDictionary(item => item.Key, item => item.Value))
            .ToList();
    }

    public List<QueryDataSet> QueryDapperMultipleRawSql(string sql)
    {
        using var conn = GetSqlConnection();
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
        using var command = CreateCommand(sql);
        using var reader = command.ExecuteReader();
        var dataSets = new List<QueryDataSet>();
        do
        {
            dataSets.Add(ReadDataSet(reader));
        } while (reader.NextResult());

        return dataSets;
    }

    public void DeleteTable(string tableName)
    {
        var sql = new StringBuilder();
        sql.Append("DROP TABLE");
        sql.Append($" [dbo].[{tableName}]");
        Execute(sql.ToString());
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
                    sql.Append("nvarchar(2000) NULL");
                    break;
            }

            if (header != headers.Last())
            {
                sql.Append(",");
            }
        }

        sql.Append(")");
        Execute(sql.ToString());
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
        return Execute(sql);
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

    private SqlConnection GetSqlConnection()
    {
        return new SqlConnection(Database.GetDbConnection().ConnectionString);
    }

    private DbCommand CreateCommand(string sql)
    {
        var conn = GetDbConnection();
        var command = conn.CreateCommand();
        command.CommandText = sql;
        command.CommandType = CommandType.Text;
        return command;
    }

    private DbConnection GetDbConnection()
    {
        var conn = Database.GetDbConnection();
        if (conn.State != ConnectionState.Open)
        {
            conn.Open();
        }

        return conn;
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

    public IEnumerable<T> Query<T>(string sql, object? parameters = null)
    {
        // var connectionString = Database.GetDbConnection().ConnectionString;
        // using var connection = new SqlConnection(connectionString);
        // connection.Open();
        var connection = GetDbConnection();
        return connection.Query<T>(sql, parameters);
    }

    public int Execute(string sql, object? parameters = null)
    {
        using var connection = new SqlConnection(Database.GetDbConnection().ConnectionString);
        connection.Open();
        return connection.Execute(sql, parameters);
    }

    public int DropTable(string tableName)
    {
        var sql = $"IF (OBJECT_ID('{tableName}')) Is Not NULL DROP TABLE [{tableName}];";
        return Execute(sql);
    }

    private static string CreateInsertTableSqlBlock(StringBuilder sqlInsertColumns, List<ExcelColumn> headers,
        List<Dictionary<string, string>> rows)
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
}

public class TableColumnInfoEntity
{
    public string Name { get; set; } = string.Empty;
    public string DataType { get; set; } = string.Empty;
    public int? Size { get; set; }
    public bool IsIdentity { get; set; }
    public bool IsNullable { get; set; }
    public int? Precision { get; set; }
    public int? Scale { get; set; }
    public bool IsKey { get; set; }
}