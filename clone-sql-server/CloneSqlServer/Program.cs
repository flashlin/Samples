using Microsoft.Data.SqlClient;
using System.Text;
using Dapper;

public class DatabaseInfo
{
    public string? Name { get; set; }
    public string? SchemaName { get; set; }
    public string? ObjectName { get; set; }
    public string? ObjectType { get; set; }
    public string? Definition { get; set; }
}

public class TableSchemaInfo
{
    public string TableName { get; set; } = string.Empty;
    public string ColumnName { get; set; } = string.Empty;
    public string DataType { get; set; } = string.Empty;
    public int? CharacterMaxLength { get; set; }
    public int? NumericPrecision { get; set; }
    public int? NumericScale { get; set; }
    public bool IsNullable { get; set; }
    public bool IsIdentity { get; set; }
}

class Program
{
    static async Task Main(string[] args)
    {
        if (args.Length < 2)
        {
            Console.WriteLine("請提供 SQL Server 連接字串和目標路徑");
            Console.WriteLine("格式：servername[:port] targetPath");
            Console.WriteLine("範例：127.0.0.1:3390 D:\\Backup\\Schema");
            return;
        }

        string connectionString = BuildConnectionString(args[0]);
        string targetPath = args[1];

        // 確保目標目錄存在
        Directory.CreateDirectory(targetPath);

        Console.WriteLine($"連接字串: {connectionString}");
        using var connection = new SqlConnection(connectionString);
        await connection.OpenAsync();
        Console.WriteLine("成功連接到 SQL Server");

        var schemaScript = await GenerateDatabaseSchemaScript(connection);
        await SaveSchemaScript(schemaScript, targetPath);
    }

    private static string BuildConnectionString(string server)
    {
        string[] serverParts = server.Split(':');
        string serverName = serverParts[0];
        string port = serverParts.Length > 1 ? $",{serverParts[1]}" : string.Empty;
        
        return $"Server={serverName}{port};Integrated Security=True;TrustServerCertificate=True;";
    }

    private static async Task<string> GenerateDatabaseSchemaScript(SqlConnection connection)
    {
        var databases = await GetUserDatabases(connection);
        var schemaScript = new StringBuilder();

        foreach (string database in databases)
        {
            await GenerateDatabaseObjects(connection, database, schemaScript);
        }

        return schemaScript.ToString();
    }

    private static async Task<IEnumerable<string>> GetUserDatabases(SqlConnection connection)
    {
        return await connection.QueryAsync<string>(
            "SELECT name FROM sys.databases WHERE database_id > 4"
        );
    }

    private static async Task GenerateDatabaseObjects(SqlConnection connection, string database, StringBuilder schemaScript)
    {
        AppendCreateDatabaseScript(schemaScript, database);

        connection.ChangeDatabase(database);
        
        schemaScript.AppendLine($"USE [{database}]");
        schemaScript.AppendLine("GO");

        await GenerateTableDefinitions(connection, schemaScript);
        //await GenerateUserFunctions(connection, schemaScript);
        //await GenerateUserDefineTypes(connection, schemaScript);
        //await GenerateViews(connection, schemaScript);
        //await GenerateStoredProcedures(connection, schemaScript);
    }

    private static void AppendCreateDatabaseScript(StringBuilder schemaScript, string database)
    {
        schemaScript.AppendLine($"IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'{database}')");
        schemaScript.AppendLine("BEGIN");
        schemaScript.AppendLine($"    CREATE DATABASE [{database}]");
        schemaScript.AppendLine("END");
        schemaScript.AppendLine("GO");
    }

    private static async Task GenerateUserFunctions(SqlConnection connection, StringBuilder schemaScript)
    {
        var query = @"
            SELECT 
                DB_NAME() as Name,
                SCHEMA_NAME(o.schema_id) as SchemaName,
                o.name as ObjectName,
                o.type as ObjectType,
                m.definition as Definition
            FROM sys.objects o
            INNER JOIN sys.sql_modules m ON o.object_id = m.object_id
            WHERE o.type IN ('FN', 'IF', 'TF', 'AF')  -- Scalar, Inline Table-valued, Table-valued, and Aggregate functions
                AND o.is_ms_shipped = 0
                AND SCHEMA_NAME(o.schema_id) != 'sys'
            ORDER BY o.type, o.name";

        var dbObjects = await connection.QueryAsync<DatabaseInfo>(query);

        foreach (var obj in dbObjects)
        {
            if (!string.IsNullOrEmpty(obj.Definition))
            {
                string functionType = obj.ObjectType switch
                {
                    "FN" => "Scalar Function",
                    "IF" => "Inline Table-valued Function",
                    "TF" => "Table-valued Function",
                    "AF" => "Aggregate Function",
                    _ => "Function"
                };

                schemaScript.AppendLine($"-- {functionType}: [{obj.SchemaName}].[{obj.ObjectName}]");
                schemaScript.AppendLine($"IF OBJECT_ID(N'[{obj.SchemaName}].[{obj.ObjectName}]') IS NOT NULL");
                schemaScript.AppendLine($"    DROP FUNCTION [{obj.SchemaName}].[{obj.ObjectName}]");
                schemaScript.AppendLine("GO");
                schemaScript.AppendLine(obj.Definition);
                schemaScript.AppendLine("GO");
                schemaScript.AppendLine();
            }
        }
    }

    private static async Task GenerateUserDefineTypes(SqlConnection connection, StringBuilder schemaScript)
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

        var dbObjects = await connection.QueryAsync<DatabaseInfo>(query);

        foreach (var obj in dbObjects)
        {
            if (!string.IsNullOrEmpty(obj.Definition))
            {
                schemaScript.AppendLine($"-- User-Defined Type: {obj.ObjectName}");
                schemaScript.AppendLine($"IF TYPE_ID(N'{obj.ObjectName}') IS NOT NULL");
                schemaScript.AppendLine($"    DROP TYPE [{obj.ObjectName}]");
                schemaScript.AppendLine("GO");
                schemaScript.AppendLine(obj.Definition);
                schemaScript.AppendLine("GO");
            }
        }
    }

    private static async Task GenerateStoredProcedures(SqlConnection connection, StringBuilder schemaScript)
    {
        var query = @"
            SELECT 
                DB_NAME() as Name,
                o.name as ObjectName,
                o.type as ObjectType,
                OBJECT_DEFINITION(o.object_id) as Definition
            FROM sys.objects o
            WHERE type in ('P')
            AND is_ms_shipped = 0";

        var dbObjects = await connection.QueryAsync<DatabaseInfo>(query);

        foreach (var obj in dbObjects)
        {
            if (!string.IsNullOrEmpty(obj.Definition))
            {
                schemaScript.AppendLine($"-- Stored Procedure: {obj.ObjectName}");
                schemaScript.AppendLine(obj.Definition);
                schemaScript.AppendLine("GO");
            }
        }
    }

    private static async Task GenerateViews(SqlConnection connection, StringBuilder schemaScript)
    {
        var query = @"
            SELECT 
                DB_NAME() as Name,
                o.name as ObjectName,
                o.type as ObjectType,
                OBJECT_DEFINITION(o.object_id) as Definition
            FROM sys.objects o
            WHERE type in ('V')
            AND is_ms_shipped = 0";

        var dbObjects = await connection.QueryAsync<DatabaseInfo>(query);

        foreach (var obj in dbObjects)
        {
            if (!string.IsNullOrEmpty(obj.Definition))
            {
                schemaScript.AppendLine($"-- View: {obj.ObjectName}");
                schemaScript.AppendLine(obj.Definition);
                schemaScript.AppendLine("GO");
            }
        }
    }

    private static async Task GenerateTableDefinitions(SqlConnection connection, StringBuilder schemaScript)
    {
        var tables = (await GetTables(connection)).ToList();
        const int batchSize = 50;

        for (int i = 0; i < tables.Count; i += batchSize)
        {
            var batch = tables.Skip(i).Take(batchSize).ToList();

            var tableSchemas = await GetBatchTableColumnDefinitions(connection, batch.Select(t => t.ObjectName!).ToList());
            foreach (var tableGroup in tableSchemas.GroupBy(t => t.TableName))
            {
                var tableName = tableGroup.Key;
                var columnDefinitions = BuildColumnDefinitions(tableGroup);

                Console.WriteLine($"Processing table {tableName}");
                AppendTableDefinition(schemaScript, tableName, columnDefinitions);
            }
        }
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

    private static string BuildColumnDefinitions(IGrouping<string, TableSchemaInfo> columns)
    {
        return string.Join(",\n    ", columns.Select(col => BuildColumnDefinition(col)));
    }

    private static string BuildColumnDefinition(TableSchemaInfo column)
    {
        var sb = new StringBuilder();
        
        sb.Append($"[{column.ColumnName}] {column.DataType}");

        // 只有特定資料類型需要加入長度/精度/小數點
        if (column.DataType.ToLower() is "varchar" or "nvarchar" or "char" or "nchar" or "binary" or "varbinary")
        {
            if (column.CharacterMaxLength.HasValue)
            {
                sb.Append(column.CharacterMaxLength == -1 ? "(MAX)" : $"({column.CharacterMaxLength})");
            }
        }
        else if (column.DataType.ToLower() is "decimal" or "numeric")
        {
            if (column.NumericPrecision.HasValue)
            {
                if (column.NumericScale.HasValue && column.NumericScale.Value > 0)
                {
                    sb.Append($"({column.NumericPrecision},{column.NumericScale})");
                }
                else
                {
                    sb.Append($"({column.NumericPrecision})");
                }
            }
        }

        // 添加 IDENTITY
        if (column.IsIdentity)
        {
            sb.Append(" IDENTITY(1,1)");
        }

        // 添加 NULL/NOT NULL
        sb.Append(column.IsNullable ? " NULL" : " NOT NULL");

        return sb.ToString();
    }

    private static void AppendTableDefinition(StringBuilder schemaScript, string tableName, string columnDefinitions)
    {
        schemaScript.AppendLine($"-- Table: {tableName}");
        schemaScript.AppendLine($"CREATE TABLE [{tableName}] (");
        schemaScript.AppendLine($"    {columnDefinitions}");
        schemaScript.AppendLine(")");
        schemaScript.AppendLine("GO");
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

    private static async Task SaveSchemaScript(string schemaScript, string targetPath)
    {
        string filePath = Path.Combine(targetPath, "CreateDatabase.sql");
        await File.WriteAllTextAsync(filePath, schemaScript);
        Console.WriteLine($"資料庫結構已成功導出到 {filePath}");
    }
}
