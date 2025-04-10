using Dapper;
using Microsoft.Data.SqlClient;

namespace CloneSqlServer;

public class GenerateContext
{
    private static readonly string[] DatabaseNameWhiteList =
    [
        "MembersInfoDB",
        "AccountDB",
    ];
    public List<string> Databases { get; set; } = new();
    public Dictionary<string, List<DatabaseInfo>> Tables { get; set; } = new();
    public Dictionary<string, List<TableSchemaInfo>> TableSchemas { get; set; } = new();
    public Dictionary<string, List<TableIndexSchema>> TableIndexes { get; set; } = new();

    public static async Task<GenerateContext> Initialize(SqlConnection connection)
    {
        var context = new GenerateContext();
        context.Databases = await GetUserDatabases(connection);
        // For now, we only use the white list
        context.Databases = DatabaseNameWhiteList.ToList();
        
        foreach (var database in context.Databases)
        {
            Console.WriteLine($"Fetch databases Schema {database}...");
            connection.ChangeDatabase(database);
            var tables = (await GetTables(connection)).ToList();
            context.Tables[database] = tables;

            var tableSchemas = new List<TableSchemaInfo>();
            const int batchSize = 50;
            
            Console.WriteLine($"Fetch Table Schemas ...");
            for (int i = 0; i < tables.Count; i += batchSize)
            {
                var batch = tables.Skip(i).Take(batchSize).ToList();
                var batchSchemas = await GetBatchTableColumnDefinitions(connection, batch.Select(t => t.ObjectName!).ToList());
                tableSchemas.AddRange(batchSchemas);
            }
            
            context.TableSchemas[database] = tableSchemas.ToList();
            context.TableIndexes[database] = await GetTablePkFkIndexs(connection, tables.Select(t => t.ObjectName!).ToList());
        }

        return context;
    }

    private static async Task<List<string>> GetUserDatabases(SqlConnection connection)
    {
        var result = await connection.QueryAsync<string>(
            "SELECT name FROM sys.databases WHERE database_id > 4"
        );
        return result.ToList();
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

    private static async Task<List<TableIndexSchema>> GetTablePkFkIndexs(SqlConnection connection, List<string> tableNames)
    {
        Console.WriteLine($"Fetch Table Indexes Schema...");
        var result = new List<TableIndexSchema>();

        // 取得所有類型的索引
        var pks = await GetPrimaryKeyIndexes(connection, tableNames);
        var fks = await GetForeignKeyIndexes(connection, tableNames);
        var indexes = await GetNormalIndexes(connection, tableNames);

        result.AddRange(pks);
        result.AddRange(fks);
        result.AddRange(indexes);

        return result;
    }

    private static async Task<IEnumerable<TableIndexSchema>> GetPrimaryKeyIndexes(SqlConnection connection, List<string> tableNames)
    {
        var pkQuery = @"
            SELECT 
                SCHEMA_NAME(t.schema_id) + '.' + t.name as TableName,
                i.name as IndexName,
                'PK' as IndexType,
                1 as IsPrimaryKey,
                i.is_unique as IsUnique,
                CASE WHEN i.type_desc LIKE '%CLUSTERED%' THEN 1 ELSE 0 END as IsClustered,
                STRING_AGG(c.name, ',') WITHIN GROUP (ORDER BY ic.key_ordinal) as ColumnsString
            FROM sys.tables t
            INNER JOIN sys.indexes i ON t.object_id = i.object_id
            INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
            INNER JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
            WHERE i.is_primary_key = 1
            AND SCHEMA_NAME(t.schema_id) + '.' + t.name IN @TableNames
            GROUP BY 
                SCHEMA_NAME(t.schema_id) + '.' + t.name,
                i.name,
                i.is_unique,
                i.type_desc";

        return await connection.QueryAsync<TableIndexSchema>(pkQuery, new { TableNames = tableNames });
    }

    private static async Task<IEnumerable<TableIndexSchema>> GetForeignKeyIndexes(SqlConnection connection, List<string> tableNames)
    {
        var fkQuery = @"
            SELECT 
                SCHEMA_NAME(t.schema_id) + '.' + t.name as TableName,
                fk.name as IndexName,
                'FK' as IndexType,
                0 as IsPrimaryKey,
                0 as IsUnique,
                0 as IsClustered,
                STRING_AGG(c.name, ',') WITHIN GROUP (ORDER BY fkc.constraint_column_id) as ColumnsString,
                SCHEMA_NAME(rt.schema_id) + '.' + rt.name as ReferencedTableName,
                STRING_AGG(rc.name, ',') WITHIN GROUP (ORDER BY fkc.constraint_column_id) as ReferencedColumnsString
            FROM sys.tables t
            INNER JOIN sys.foreign_keys fk ON t.object_id = fk.parent_object_id
            INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
            INNER JOIN sys.columns c ON fkc.parent_object_id = c.object_id AND fkc.parent_column_id = c.column_id
            INNER JOIN sys.tables rt ON fk.referenced_object_id = rt.object_id
            INNER JOIN sys.columns rc ON fkc.referenced_object_id = rc.object_id AND fkc.referenced_column_id = rc.column_id
            WHERE SCHEMA_NAME(t.schema_id) + '.' + t.name IN @TableNames
            GROUP BY 
                SCHEMA_NAME(t.schema_id) + '.' + t.name,
                fk.name,
                SCHEMA_NAME(rt.schema_id) + '.' + rt.name";

        return await connection.QueryAsync<TableIndexSchema>(fkQuery, new { TableNames = tableNames });
    }

    private static async Task<IEnumerable<TableIndexSchema>> GetNormalIndexes(SqlConnection connection, List<string> tableNames)
    {
        var indexQuery = @"
            SELECT 
                SCHEMA_NAME(t.schema_id) + '.' + t.name as TableName,
                i.name as IndexName,
                'INDEX' as IndexType,
                0 as IsPrimaryKey,
                i.is_unique as IsUnique,
                CASE WHEN i.type_desc LIKE '%CLUSTERED%' THEN 1 ELSE 0 END as IsClustered,
                STRING_AGG(c.name, ',') WITHIN GROUP (ORDER BY ic.key_ordinal) as ColumnsString
            FROM sys.tables t
            INNER JOIN sys.indexes i ON t.object_id = i.object_id
            INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
            INNER JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
            WHERE i.is_primary_key = 0 
            AND i.is_unique_constraint = 0
            AND i.type > 0  -- 排除 Heap
            AND SCHEMA_NAME(t.schema_id) + '.' + t.name IN @TableNames
            GROUP BY 
                SCHEMA_NAME(t.schema_id) + '.' + t.name,
                i.name,
                i.is_unique,
                i.type_desc";

        return await connection.QueryAsync<TableIndexSchema>(indexQuery, new { TableNames = tableNames });
    }
}