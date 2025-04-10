using Dapper;
using Microsoft.Data.SqlClient;

namespace CloneSqlServer;

public class LoginRoleInfo
{
    public string LoginName { get; set; } = string.Empty;
    public string RoleName { get; set; } = string.Empty;
}

public class GenerateContext
{
    private static readonly string[] DatabaseNameWhiteList =
    [
        "MembersInfoDB",
        "AccountDB",
        "PlutoRepSB",
        "CashmarketDB",
        "MailManagement",
        "PromotionManagement",
        "PromotionManagementHistory"
    ];
    public List<string> Databases { get; set; } = new();
    public Dictionary<string, List<DatabaseInfo>> Tables { get; set; } = new();
    public Dictionary<string, List<TableSchemaInfo>> TableSchemas { get; set; } = new();
    public Dictionary<string, List<TableIndexSchema>> TableIndexes { get; set; } = new();
    public List<string> LoginNames { get; set; } = new();
    public List<LoginRoleInfo> LoginRoles { get; set; } = [];

    /// <summary>
    /// 取得所有 SQL Server 登入帳號（僅 SQL Login）
    /// </summary>
    /// <param name="connection">資料庫連線</param>
    /// <returns>登入帳號清單</returns>
    private static async Task<List<string>> GetAllLoginNames(SqlConnection connection)
    {
        var query = @"
            SELECT name 
            FROM sys.server_principals 
            WHERE type_desc = 'SQL_LOGIN'
            AND name NOT LIKE '##%'
            AND name != 'sa'
            ORDER BY name";

        var result = await connection.QueryAsync<string>(query);
        return result.ToList();
    }

    /// <summary>
    /// 取得指定登入帳號的所有角色
    /// </summary>
    /// <param name="connection">資料庫連線</param>
    /// <param name="loginNames">登入帳號清單</param>
    /// <returns>登入帳號和角色的對應清單</returns>
    private static async Task<List<LoginRoleInfo>> GetAllUserRoles(SqlConnection connection, List<string> loginNames)
    {
        if (loginNames == null || !loginNames.Any())
            return new List<LoginRoleInfo>();

        var query = @"
            SELECT 
                sp.name AS LoginName,
                srm.role_principal_id,
                ISNULL(sp2.name, 'public') AS RoleName
            FROM sys.server_principals sp
            LEFT JOIN sys.server_role_members srm 
                ON sp.principal_id = srm.member_principal_id
            LEFT JOIN sys.server_principals sp2 
                ON srm.role_principal_id = sp2.principal_id
            WHERE sp.type_desc = 'SQL_LOGIN'
                AND sp.name IN @LoginNames
            ORDER BY sp.name, sp2.name";

        var result = await connection.QueryAsync<LoginRoleInfo>(query, new { LoginNames = loginNames });
        return result.ToList();
    }

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

        Console.WriteLine($"Fetch SQL Logins ...");
        context.LoginNames = await GetAllLoginNames(connection);
        context.LoginRoles = await GetAllUserRoles(connection, context.LoginNames);
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