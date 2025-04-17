using System.Text.Json;
using Dapper;
using Microsoft.Data.SqlClient;

namespace CloneSqlServer;

public class StoreProcedureInfo
{
    public string DatabaseName { get; set; } = string.Empty;
    public string StoreProcedureName { get; set; } = string.Empty;
    public string Definition { get; set; } = string.Empty;
}

public class LoginRoleInfo
{
    public string DatabaseName { get; set; } = string.Empty;
    public string LoginName { get; set; } = string.Empty;
    public string RoleName { get; set; } = string.Empty;
}

public class StoreProcedurePermissionInfo
{
    public string DatabaseName { get; set; } = string.Empty;
    public string RoleName { get; set; } = string.Empty;
    public string StoreProcedureName { get; set; } = string.Empty;
    public string ObjectType { get; set; } = string.Empty;
    public string PermissionName { get; set; } = string.Empty;
}

public class UserDefinedTypeInfo
{
    public string Name { get; set; } = string.Empty;
    public string ObjectName { get; set; } = string.Empty;
    public string ObjectType { get; set; } = string.Empty;
    public string Definition { get; set; } = string.Empty;
}

public class UserDefinedTypeWithRoleInfo
{
    public string DatabaseName { get; set; } = string.Empty;
    public string TypeName { get; set; } = string.Empty;
    public string Definition { get; set; } = string.Empty;
    public string RoleNames { get; set; } = string.Empty;
    public List<string> RoleNamesList 
    { 
        get 
        {
            return string.IsNullOrEmpty(RoleNames) 
                ? new List<string>() 
                : RoleNames.Split(',').Select(x => x.Trim()).ToList();
        }
    }
}

public class LoginNameInfo
{
    public string DatabaseName { get; set; } = string.Empty;
    public string LoginName { get; set; } = string.Empty;
}

public class DatabaseRoleInfo
{
    public string DatabaseName { get; set; } = string.Empty;
    public string RoleName { get; set; } = string.Empty;
}

public class ConstraintsInfo
{
    public string DatabaseName { get; set; } = string.Empty;
    public string TableName { get; set; } = string.Empty;
    public string ConstraintName { get; set; } = string.Empty;
    public string ConstraintDefine { get; set; } = string.Empty;
    public string ColumnName { get; set; } = string.Empty;
}

public class GenerateContext
{
    public static readonly string[] DatabaseNameWhiteList =
    [
    ];
    public List<string> Databases { get; set; } = new();
    public Dictionary<string, List<DatabaseInfo>> Tables { get; set; } = new();
    public Dictionary<string, List<TableSchemaInfo>> TableSchemas { get; set; } = new();
    public Dictionary<string, List<TableIndexSchema>> TableIndexes { get; set; } = new();
    public List<LoginNameInfo> LoginNames { get; set; } = [];
    public List<LoginRoleInfo> LoginRoles { get; set; } = [];
    public List<DatabaseRoleInfo> DatabaseRoleNames { get; set; } = [];
    public List<StoreProcedurePermissionInfo> StoreProcedurePermissions { get; set; } = [];
    public List<StoreProcedureInfo> StoreProcedures { get; set; } = [];
    public List<StoreProcedureInfo> IndependentStoreProcedures { get; set; } = [];
    public List<UserDefinedTypeInfo> UserDefinedTypes { get; set; } = [];
    public List<UserDefinedTypeWithRoleInfo> UserDefinedTypeWithRoles { get; set; } = [];
    public List<ConstraintsInfo> TableConstraints { get; set; } = [];
    

    /// <summary>
    /// Get all SQL Server login accounts (SQL Login only)
    /// </summary>
    /// <param name="connection">Database connection</param>
    /// <returns>List of login accounts with database name</returns>
    private static async Task<List<LoginNameInfo>> GetAllLoginNames(SqlConnection connection)
    {
        var query = @"
            SELECT 
                DB_NAME() as DatabaseName,
                name as LoginName
            FROM sys.server_principals 
            WHERE type_desc = 'SQL_LOGIN'
            AND name NOT LIKE '##%'
            AND name != 'sa'
            ORDER BY name";

        var result = await connection.QueryAsync<LoginNameInfo>(query);
        return result.ToList();
    }

    /// <summary>
    /// 取得指定登入帳號的所有資料庫角色
    /// </summary>
    /// <param name="connection">資料庫連線</param>
    /// <param name="loginNames">登入帳號清單</param>
    /// <returns>登入帳號和資料庫角色的對應清單</returns>
    private static async Task<List<LoginRoleInfo>> GetAllUserRoles(SqlConnection connection, List<string> loginNames)
    {
        if (loginNames == null || !loginNames.Any())
            return new List<LoginRoleInfo>();

        var query = @"
            SELECT 
                DB_NAME() AS DatabaseName,
                dp.name AS LoginName,
                ISNULL(rp.name, 'public') AS RoleName
            FROM sys.database_principals dp
            LEFT JOIN sys.database_role_members drm ON dp.principal_id = drm.member_principal_id
            LEFT JOIN sys.database_principals rp ON drm.role_principal_id = rp.principal_id
            WHERE dp.authentication_type_desc = 'INSTANCE'
              AND dp.sid IN (
                  SELECT sid FROM sys.server_principals WHERE name IN @LoginNames
              )
            ORDER BY dp.name, rp.name";

        var result = await connection.QueryAsync<LoginRoleInfo>(query, new { LoginNames = loginNames });
        return result.ToList();
    }

    /// <summary>
    /// 取得所有資料庫角色
    /// </summary>
    /// <param name="connection">資料庫連線</param>
    /// <returns>資料庫角色清單</returns>
    private static async Task<List<DatabaseRoleInfo>> GetAllDatabaseRoles(SqlConnection connection)
    {
        var query = @"
            SELECT 
                DB_NAME() AS DatabaseName,
                name AS RoleName
            FROM sys.database_principals 
            WHERE type = 'R'
            AND is_fixed_role = 0
            AND name NOT LIKE '##%'
            ORDER BY name";

        var result = await connection.QueryAsync<DatabaseRoleInfo>(query);
        return result.ToList();
    }

    /// <summary>
    /// 取得所有預存程序的權限，且只包含獨立預存程序的權限
    /// </summary>
    /// <param name="connection">資料庫連線</param>
    /// <returns>權限清單</returns>
    private static async Task<List<StoreProcedurePermissionInfo>> GetIndependentStoreProcedurePermissions(SqlConnection connection)
    {
        // 先取得所有獨立預存程序
        var independentSps = await GetIndependentStoreProcedures(connection);
        var independentSpKeys = independentSps
            .Select(sp => $"{sp.DatabaseName}.{sp.StoreProcedureName}")
            .ToHashSet();

        var query = @"
            SELECT 
                DB_NAME() AS DatabaseName,
                dp.name AS RoleName,
                o.name AS StoreProcedureName,
                o.type_desc AS ObjectType,
                perm.permission_name AS PermissionName
            FROM sys.database_permissions perm
            JOIN sys.objects o ON perm.major_id = o.object_id
            JOIN sys.database_principals dp ON perm.grantee_principal_id = dp.principal_id
            WHERE 
                o.type = 'P' -- Stored Procedure
                AND perm.permission_name IN ('EXECUTE')";

        var result = await connection.QueryAsync<StoreProcedurePermissionInfo>(query);
        
        // 在記憶體中進行 join，只保留獨立預存程序的權限
        return result
            .Where(p => independentSpKeys.Contains($"{p.DatabaseName}.{p.StoreProcedureName}"))
            .ToList();
    }

    public static async Task<GenerateContext> Initialize(SqlConnection connection)
    {
        var context = new GenerateContext();
        context.Databases = await GetUserDatabases(connection);
        // For now, we only use the white list
        //context.Databases = DatabaseNameWhiteList.ToList();
        
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
            
            Console.WriteLine($"Fetch Store Procedures ...");
            context.StoreProcedures.AddRange((await GetAllStoreProceduresByDatabase(connection)).ToList());
            
            Console.WriteLine($"Fetch Independent Store Procedures ...");
            context.IndependentStoreProcedures.AddRange(await GetIndependentStoreProcedures(connection));

            Console.WriteLine($"Fetch User Defined Types ...");
            context.UserDefinedTypes.AddRange(await GetUserDefinedTypes(connection));

            Console.WriteLine($"Fetch User Defined Type With Roles ...");
            context.UserDefinedTypeWithRoles.AddRange(await GetUserDefinedTypeWithRoles(connection));
            
            Console.WriteLine($"Fetch SQL Logins for {database}...");
            var loginNames = await GetAllLoginNames(connection);
            context.LoginNames.AddRange(loginNames);
            
            var loginNamesList = loginNames.Select(r => r.LoginName).ToList();
            context.LoginRoles.AddRange(await GetAllUserRoles(connection, loginNamesList));

            Console.WriteLine($"Fetch Database Roles for {database}...");
            context.DatabaseRoleNames.AddRange(await GetAllDatabaseRoles(connection));

            Console.WriteLine($"Fetch Independent Store Procedure Permissions for {database}...");
            context.StoreProcedurePermissions.AddRange(await GetIndependentStoreProcedurePermissions(connection));

            Console.WriteLine($"Fetch Table Constraints for {database}...");
            var tableNames = tables.Select(t => t.ObjectName!).ToList();
            context.TableConstraints.AddRange(await GetTableConstraints(connection, tableNames));
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
                t.name as ObjectName,
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
                t.name as TableName,
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
            WHERE t.name IN @TableNames
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
                t.name as TableName,
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
            AND t.name IN @TableNames
            GROUP BY 
                t.name,
                i.name,
                i.is_unique,
                i.type_desc";

        return await connection.QueryAsync<TableIndexSchema>(pkQuery, new { TableNames = tableNames });
    }

    private static async Task<IEnumerable<TableIndexSchema>> GetForeignKeyIndexes(SqlConnection connection, List<string> tableNames)
    {
        var fkQuery = @"
            SELECT 
                t.name as TableName,
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
            WHERE t.name IN @TableNames
            GROUP BY 
                t.name,
                fk.name,
                SCHEMA_NAME(rt.schema_id) + '.' + rt.name";

        return await connection.QueryAsync<TableIndexSchema>(fkQuery, new { TableNames = tableNames });
    }

    private static async Task<IEnumerable<TableIndexSchema>> GetNormalIndexes(SqlConnection connection, List<string> tableNames)
    {
        var indexQuery = @"
            SELECT 
                t.name as TableName,
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
            AND t.name IN @TableNames
            GROUP BY 
                t.name,
                i.name,
                i.is_unique,
                i.type_desc";

        return await connection.QueryAsync<TableIndexSchema>(indexQuery, new { TableNames = tableNames });
    }

    public static async Task<List<StoreProcedureInfo>> GetAllStoreProceduresByDatabase(SqlConnection connection)
    {
        var query = @"
            SELECT 
                DB_NAME() AS DatabaseName,
                o.name AS StoreProcedureName,
                OBJECT_DEFINITION(o.object_id) AS Definition
            FROM sys.objects o
            WHERE o.type = 'P'
              AND o.is_ms_shipped = 0
              AND OBJECT_DEFINITION(o.object_id) IS NOT NULL";

        var result = await connection.QueryAsync<StoreProcedureInfo>(query);
        return result.ToList();
    }

    /// <summary>
    /// Get all stored procedures without dependencies
    /// Excludes:
    /// 1. Procedures that depend on other stored procedures
    /// 2. Procedures using EXEC or sp_executesql
    /// 3. Procedures using OPENQUERY, OPENROWSET, OPENDATASOURCE
    /// </summary>
    /// <param name="connection">Database connection</param>
    /// <returns>List of independent stored procedures</returns>
    public static async Task<List<StoreProcedureInfo>> GetIndependentStoreProcedures(SqlConnection connection)
    {
        var query = @"
            SELECT 
                DB_NAME() AS DatabaseName,
                o.name AS StoreProcedureName,
                OBJECT_DEFINITION(o.object_id) AS Definition
            FROM sys.objects o
            WHERE o.type = 'P'
              AND o.is_ms_shipped = 0
              AND NOT EXISTS (
                  SELECT 1
                  FROM sys.sql_expression_dependencies d
                  WHERE d.referencing_id = o.object_id
                    AND (
                        d.referenced_id IN (
                            SELECT object_id 
                            FROM sys.objects 
                            WHERE type = 'P'
                        )
                        OR d.referenced_server_name IS NOT NULL
                    )
              )
              AND OBJECT_DEFINITION(o.object_id) NOT LIKE '%EXEC %'
              AND OBJECT_DEFINITION(o.object_id) NOT LIKE '%sp_executesql%'
              AND OBJECT_DEFINITION(o.object_id) NOT LIKE '%OPENQUERY%'
              AND OBJECT_DEFINITION(o.object_id) NOT LIKE '%OPENROWSET%'
              AND OBJECT_DEFINITION(o.object_id) NOT LIKE '%OPENDATASOURCE%'";

        var result = await connection.QueryAsync<StoreProcedureInfo>(query);
        return result.ToList();
    }

    /// <summary>
    /// Get all user-defined types including table types, scalar types, etc.
    /// </summary>
    /// <param name="connection">Database connection</param>
    /// <returns>List of user-defined types with their definitions</returns>
    private static async Task<List<UserDefinedTypeInfo>> GetUserDefinedTypes(SqlConnection connection)
    {
        var query = @"
            SELECT 
                DB_NAME() as Name,
                t.name as ObjectName,
                'UDT' as ObjectType,
                CASE 
                    WHEN t.is_table_type = 1 THEN
                    (
                        SELECT 
                            'CREATE TYPE [' + SCHEMA_NAME(tt.schema_id) + '].[' + tt.name + '] AS TABLE (' +
                            STUFF((
                                SELECT ', [' + c.name + '] ' + 
                                    tp.name + 
                                    CASE 
                                        WHEN tp.name IN ('varchar', 'nvarchar', 'char', 'nchar') 
                                            THEN '(' + CASE WHEN c.max_length = -1 
                                                THEN 'MAX' 
                                                ELSE CAST(CASE WHEN tp.name LIKE 'n%' 
                                                    THEN c.max_length/2 
                                                    ELSE c.max_length END AS VARCHAR) 
                                            END + ')'
                                        WHEN tp.name IN ('decimal', 'numeric') 
                                            THEN '(' + CAST(c.[precision] AS VARCHAR) + ',' + CAST(c.scale AS VARCHAR) + ')'
                                        WHEN tp.name IN ('binary', 'varbinary')
                                            THEN '(' + CASE WHEN c.max_length = -1 
                                                THEN 'MAX' 
                                                ELSE CAST(c.max_length AS VARCHAR) 
                                            END + ')'
                                        ELSE ''
                                    END +
                                    CASE WHEN c.is_nullable = 1 THEN ' NULL' ELSE ' NOT NULL' END
                                FROM sys.table_types tt2
                                INNER JOIN sys.columns c ON c.object_id = tt2.type_table_object_id
                                INNER JOIN sys.types tp ON c.user_type_id = tp.user_type_id
                                WHERE tt2.user_type_id = t.user_type_id
                                ORDER BY c.column_id
                                FOR XML PATH(''), TYPE).value('.', 'nvarchar(max)'), 1, 2, '') + ')'
                        FROM sys.table_types tt
                        WHERE tt.user_type_id = t.user_type_id
                    )
                    ELSE 
                        'CREATE TYPE [' + SCHEMA_NAME(t.schema_id) + '].[' + t.name + '] FROM ' +
                        CASE 
                            WHEN t.is_table_type = 0 THEN
                                base_type.name +
                                CASE 
                                    WHEN t.max_length = -1 THEN '(MAX)'
                                    WHEN t.max_length > 0 AND base_type.name IN ('varchar', 'nvarchar', 'char', 'nchar', 'binary', 'varbinary')
                                        THEN '(' + CAST(t.max_length AS VARCHAR) + ')'
                                    WHEN base_type.name IN ('decimal', 'numeric')
                                        THEN '(' + CAST(t.precision AS VARCHAR) + ',' + CAST(t.scale AS VARCHAR) + ')'
                                    ELSE ''
                                END
                            ELSE ''
                        END
                END as Definition
            FROM sys.types t
            LEFT JOIN sys.types base_type ON t.system_type_id = base_type.user_type_id
            WHERE t.is_user_defined = 1
                AND t.schema_id <> SCHEMA_ID('sys')
            ORDER BY t.name";

        var result = await connection.QueryAsync<UserDefinedTypeInfo>(query);
        return result.ToList();
    }

    /// <summary>
    /// Get user defined types with their role permissions
    /// </summary>
    /// <returns>List of user defined types with role permissions</returns>
    private static async Task<List<UserDefinedTypeWithRoleInfo>> GetUserDefinedTypeWithRoles(SqlConnection connection)
    {
        var result = new List<UserDefinedTypeWithRoleInfo>();

        var query = @"
            SELECT 
                DB_NAME() as DatabaseName,
                t.name as TypeName,
                OBJECT_DEFINITION(t.user_type_id) as Definition,
                STRING_AGG(dp.name, ',') as RoleNames
            FROM sys.types t
            LEFT JOIN sys.database_permissions p 
                ON p.major_id = t.user_type_id 
                AND p.permission_name = 'EXECUTE'
            LEFT JOIN sys.database_principals dp 
                ON p.grantee_principal_id = dp.principal_id
            WHERE t.is_user_defined = 1
                AND t.schema_id <> SCHEMA_ID('sys')
            GROUP BY t.user_type_id, t.name
            ORDER BY t.name";

        var udtWithRoles = await connection.QueryAsync<UserDefinedTypeWithRoleInfo>(query);
        result.AddRange(udtWithRoles);

        return result;
    }

    /// <summary>
    /// Get all table constraints for the specified tables
    /// </summary>
    /// <param name="connection">Database connection</param>
    /// <param name="tableNames">List of table names to get constraints for</param>
    /// <returns>List of table constraints</returns>
    private static async Task<List<ConstraintsInfo>> GetTableConstraints(SqlConnection connection, List<string> tableNames)
    {
        var query = @"
            SELECT 
                DB_NAME() AS DatabaseName,
                t.name AS TableName,
                dc.name AS ConstraintName,
                dc.definition AS ConstraintDefine,
                c.name AS ColumnName
            FROM sys.tables t
            INNER JOIN sys.default_constraints dc ON t.object_id = dc.parent_object_id
            INNER JOIN sys.columns c ON dc.parent_object_id = c.object_id AND dc.parent_column_id = c.column_id
            WHERE t.name IN @TableNames
            UNION ALL
            SELECT 
                DB_NAME() AS DatabaseName,
                t.name AS TableName,
                cc.name AS ConstraintName,
                cc.definition AS ConstraintDefine,
                '' AS ColumnName
            FROM sys.tables t
            INNER JOIN sys.check_constraints cc ON t.object_id = cc.parent_object_id
            WHERE t.name IN @TableNames
            ORDER BY TableName, ConstraintName";

        var result = await connection.QueryAsync<ConstraintsInfo>(query, new { TableNames = tableNames });
        return result.ToList();
    }
}