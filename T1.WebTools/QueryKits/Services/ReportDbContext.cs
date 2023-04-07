using System.Data;
using System.Data.Common;
using System.Text;
using Dapper;
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
        var sql = SqlBuilder.CreateTableStatement(entityType);
        ExecuteRawSql(sql);
    }

    public void MergeTable(MergeTableRequest req)
    {
        var leftColumns = string.Join(",", req.LeftTable.Columns.Select(x => $"tb1.[{x.Name}]"));
        var rightColumns = string.Join(",", req.RightTable.Columns.Select(x => $"tb2.[{x.Name}]"));
        var leftJoinKeys = JoinKeys(req.LeftJoinKeys, name => $"tb1.{name}");
        var rightJoinKeys = JoinKeys(req.RightJoinKeys, name => $"tb2.{name}");

        var columnNames = GetAllColumnNames(req);
        var leftAliasNames = GetAliasNames(columnNames, req.LeftTable);
        var rightAlias = GetAliasNames(columnNames, req.RightTable);
        
        var sql = new StringBuilder();
        sql.Append("SELECT ");
        sql.Append(leftColumns + ",");
        sql.AppendLine(rightColumns);
        sql.AppendLine($"INTO {req.TargetTableName}");
        sql.AppendLine($"FROM {req.LeftTable.Name} as tb1");
        sql.Append($"JOIN {req.RightTable.Name} as tb2 ON {leftJoinKeys} = {rightJoinKeys}");
        
        

        ExecuteRawSql(sql.ToString());
    }

    private static List<string> GetAliasNames(Dictionary<string, int> columnNames, TableInfo table)
    {
        var aliasNames = new List<string>();
        foreach (var column in table.Columns)
        {
            columnNames.TryGetValue(column.Name, out var times);
            if (times > 1)
            {
                aliasNames.Add($"[{table.Name}_{column.Name}]");
            }
            else
            {
                aliasNames.Add($"[{column.Name}]");
            }
        }
        return aliasNames;
    }

    private static Dictionary<string, int> GetAllColumnNames(MergeTableRequest req)
    {
        var columnNames = new Dictionary<string, int>();
        foreach (var column in req.LeftTable.Columns)
        {
            if (columnNames.TryAdd(column.Name, 1))
            {
                columnNames[column.Name]++;
            }
        }

        foreach (var column in req.RightTable.Columns)
        {
            if (columnNames.TryAdd(column.Name, 1))
            {
                columnNames[column.Name]++;
            }
        }

        return columnNames;
    }

    private string JoinKeys(List<TableColumnInfo> joinKeys, Func<string, string> mapField)
    {
        if (joinKeys.Count == 1)
        {
            return JoinKeyName(joinKeys[0], mapField);
        }
        return "CONCAT(" +
               string.Join(",", joinKeys.Select(x => JoinKeyName(x, mapField))) + 
               ")";
    }

    private string JoinKeyName(TableColumnInfo column, Func<string, string> mapField)
    {
        string Field()
        {
            return mapField($"[{column.Name}]");
        }
        if (column.DataType.TypeName == "DATETIME")
        {
            return $"CONVERT(varchar, {Field()}, 126)";
        }
        return Field();
    }

    public List<string> GetAllTableNames()
    {
        var sql = SqlBuilder.GetAllTableNames(DatabaseName);
        return Database.SqlQueryRaw<string>(sql).ToList();
    }

    public List<Dictionary<string, object>> QueryRawSql(string sql)
    {
        using var conn = Database.GetDbConnection();
        return conn.Query(sql)
            .Cast<IDictionary<string, object>>()
            .Select(row => row.ToDictionary(item => item.Key, item => item.Value))
            .ToList();
    }

    public List<T> Query<T>(string sql, object? parameters = null)
    {
        using var conn = Database.GetDbConnection();
        return conn.Query<T>(sql, parameters)
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
        using var command = CreateCommand(sql);
        using var reader = command.ExecuteReader();
        var dataSets = new List<QueryDataSet>();
        do
        {
            dataSets.Add(ReadDataSet(reader));
        } while (reader.NextResult());

        return dataSets;
    }

    private DbCommand CreateCommand(string sql)
    {
        var conn = GetDbConnection();
        var command = conn.CreateCommand();
        command.CommandText = sql;
        command.CommandType = CommandType.Text;
        return command;
    }

    public void DeleteTable(string tableName)
    {
        var sql = new StringBuilder();
        sql.Append("DROP TABLE");
        sql.Append($" [dbo].[{tableName}]");
        ExecuteRawSql(sql.ToString());
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

    public List<TableColumnInfo> GetTableColumns(string tableName)
    {
        var sql = new StringBuilder();
        sql.Append("SELECT COLUMN_NAME as Name,");
        sql.Append("DATA_TYPE as DataType,");
        sql.Append("CHARACTER_MAXIMUM_LENGTH as Size,");
        sql.Append("NUMERIC_PRECISION as Precision,");
        sql.Append("NUMERIC_SCALE as Scale ");
        sql.Append("FROM INFORMATION_SCHEMA.COLUMNS ");
        sql.Append($"WHERE TABLE_NAME = '{tableName}'");

        return Query<TableColumnInfo>(sql.ToString());
    }

    public int ExecuteRawSql(string sql, object? parameters=null)
    {
        var conn = Database.GetDbConnection();
        return conn.Execute(sql, parameters);
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
                    sql.Append("nvarchar(2000) NULL");
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