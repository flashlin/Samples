using System.Data.SqlClient;
using Dapper;

namespace CloneDatabase;

public class SqlDb
{
    private readonly string _connectionString;

    public SqlDb(string connectionString)
    {
        _connectionString = connectionString;
    }

    public List<DatabaseInfo> QueryDatabases()
    {
        return Query<DatabaseInfo>("use [master];SELECT name, database_id as Id, create_date as CreateDate FROM sys.databases").ToList();
    }

    public List<TableInfo> QueryTables(string dbName)
    {
        return Query<TableInfo>(@$"use [{dbName}];
SELECT info.TABLE_NAME as Name
            FROM {dbName}.INFORMATION_SCHEMA.TABLES as info WITH(NOLOCK)
            WHERE TABLE_TYPE = 'BASE TABLE';").ToList();
    }

    public List<TableFieldInfo> QueryFields(string dbName, string tableName)
    {
        var sql = $@"use [{dbName}];
SELECT 
    c.name as Name,
    t.Name as DataType,
    c.max_length as MaxLength,
    c.precision ,
    c.scale,
    c.is_nullable as IsNullable,
    ISNULL(i.is_primary_key, 0) as IsPrimaryKey
FROM    
    sys.columns c
INNER JOIN 
    sys.types t ON c.user_type_id = t.user_type_id
LEFT OUTER JOIN 
    sys.index_columns ic ON ic.object_id = c.object_id AND ic.column_id = c.column_id
LEFT OUTER JOIN 
    sys.indexes i ON ic.object_id = i.object_id AND ic.index_id = i.index_id
WHERE
    c.object_id = OBJECT_ID('{tableName}')";
        return Query<TableFieldInfo>(sql).ToList();
    }

    public List<T> Query<T>(string sql, object? dbParameter = null)
    {
        using var conn = new SqlConnection(_connectionString);
        return conn.Query<T>(sql, dbParameter).ToList();
    }
}