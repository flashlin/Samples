using Microsoft.Data.SqlClient;
using Dapper;

public class GenerateContext
{
    public List<string> Databases { get; set; } = new();
    public Dictionary<string, List<DatabaseInfo>> Tables { get; set; } = new();
    public Dictionary<string, List<TableSchemaInfo>> TableSchemas { get; set; } = new();

    public static async Task<GenerateContext> Initialize(SqlConnection connection)
    {
        var context = new GenerateContext();
        context.Databases = (await GetUserDatabases(connection)).ToList();
        
        foreach (var database in context.Databases)
        {
            connection.ChangeDatabase(database);
            var tables = (await GetTables(connection)).ToList();
            context.Tables[database] = tables;

            var tableSchemas = new List<TableSchemaInfo>();
            const int batchSize = 50;
            
            for (int i = 0; i < tables.Count; i += batchSize)
            {
                var batch = tables.Skip(i).Take(batchSize).ToList();
                var batchSchemas = await GetBatchTableColumnDefinitions(connection, batch.Select(t => t.ObjectName!).ToList());
                tableSchemas.AddRange(batchSchemas);
            }
            
            context.TableSchemas[database] = tableSchemas.ToList();
        }

        return context;
    }

    private static async Task<IEnumerable<string>> GetUserDatabases(SqlConnection connection)
    {
        return await connection.QueryAsync<string>(
            "SELECT name FROM sys.databases WHERE database_id > 4"
        );
    }

    private static async Task<IEnumerable<DatabaseInfo>> GetTables(SqlConnection connection)
    {
        var tableQuery = @"
            SELECT 
                DB_NAME() as Name,
                SCHEMA_NAME(t.schema_id) + '.' + t.name as ObjectName,
                'U' as ObjectType,
                OBJECT_DEFINITION(t.object_id) as Definition
            FROM sys.tables t
            WHERE t.is_ms_shipped = 0
                AND t.type = 'U'  -- 只取得使用者定義的資料表
                AND t.temporal_type = 0  -- 排除系統版本控制的資料表
                AND SCHEMA_NAME(t.schema_id) != 'sys'  -- 排除系統結構描述
                AND t.name NOT LIKE 'dt%'  -- 排除暫存資料表
                AND t.name NOT LIKE '#%'   -- 排除暫存資料表
            ORDER BY SCHEMA_NAME(t.schema_id), t.name";

        return await connection.QueryAsync<DatabaseInfo>(tableQuery);
    }

    private static async Task<IEnumerable<TableSchemaInfo>> GetBatchTableColumnDefinitions(SqlConnection connection, List<string> tableNames)
    {
        var query = @"
            SELECT 
                SCHEMA_NAME(t.schema_id) + '.' + t.name as TableName,
                c.name as ColumnName,
                tp.name as DataType,
                CASE 
                    WHEN tp.name IN ('nchar', 'nvarchar') AND c.max_length != -1 THEN c.max_length/2
                    ELSE c.max_length
                END as CharacterMaxLength,
                c.precision as NumericPrecision,
                c.scale as NumericScale,
                c.is_nullable as IsNullable,
                c.is_identity as IsIdentity
            FROM sys.tables t
            INNER JOIN sys.columns c ON t.object_id = c.object_id
            INNER JOIN sys.types tp ON c.user_type_id = tp.user_type_id
            WHERE SCHEMA_NAME(t.schema_id) + '.' + t.name IN @TableNames
            ORDER BY 
                SCHEMA_NAME(t.schema_id) + '.' + t.name,
                c.column_id";

        return await connection.QueryAsync<TableSchemaInfo>(query, new { TableNames = tableNames });
    }
}
