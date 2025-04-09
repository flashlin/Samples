using Microsoft.Data.SqlClient;
using System.Text;
using Dapper;

public class DatabaseInfo
{
    public string? Name { get; set; }
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
        await GenerateViews(connection, schemaScript);
        await GenerateUserDefineTypes(connection, schemaScript);
        await GenerateStoredProcedures(connection, schemaScript);
    }

    private static void AppendCreateDatabaseScript(StringBuilder schemaScript, string database)
    {
        schemaScript.AppendLine($"IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'{database}')");
        schemaScript.AppendLine("BEGIN");
        schemaScript.AppendLine($"    CREATE DATABASE [{database}]");
        schemaScript.AppendLine("END");
        schemaScript.AppendLine("GO");
    }

    private static async Task GenerateUserDefineTypes(SqlConnection connection, StringBuilder schemaScript)
    {
        var query = @"
            SELECT 
                DB_NAME() as Name,
                t.name as ObjectName,
                'UDT' as ObjectType,
                CONCAT('CREATE TYPE [', SCHEMA_NAME(t.schema_id), '].[', t.name, '] FROM ', 
                    CASE WHEN t.is_table_type = 1 
                        THEN 'AS TABLE (' + 
                            (SELECT STRING_AGG(
                                CONCAT('[', c.name, '] ', 
                                    tp.name, 
                                    CASE 
                                        WHEN tp.name IN ('varchar', 'nvarchar', 'char', 'nchar') 
                                            THEN '(' + CASE WHEN c.max_length = -1 
                                                THEN 'MAX' 
                                                ELSE CAST(CASE WHEN tp.name LIKE 'n%' 
                                                    THEN c.max_length/2 
                                                    ELSE c.max_length END AS VARCHAR) 
                                            END + ')'
                                        WHEN tp.name IN ('decimal', 'numeric') 
                                            THEN '(' + CAST(c.precision AS VARCHAR) + ',' + CAST(c.scale AS VARCHAR) + ')'
                                        ELSE ''
                                    END,
                                    CASE WHEN c.is_nullable = 1 THEN ' NULL' ELSE ' NOT NULL' END
                                ), ', ')
                            FROM sys.columns c
                            INNER JOIN sys.types tp ON c.user_type_id = tp.user_type_id
                            WHERE c.object_id = t.type_table_object_id
                            ) + ')'
                        ELSE CONCAT(
                            base_type.name,
                            CASE 
                                WHEN t.max_length = -1 THEN '(MAX)'
                                WHEN t.max_length > 0 THEN '(' + CAST(t.max_length AS VARCHAR) + ')'
                                ELSE ''
                            END
                        )
                    END
                ) as Definition
            FROM sys.types t
            LEFT JOIN sys.types base_type ON t.system_type_id = base_type.user_type_id
            WHERE t.is_user_defined = 1
                AND t.schema_id <> SCHEMA_ID('sys')";

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
            Console.WriteLine($"Processing tables {i + 1} to {Math.Min(i + batchSize, tables.Count)} of {tables.Count}");

            var tableSchemas = await GetBatchTableColumnDefinitions(connection, batch.Select(t => t.ObjectName!).ToList());
            
            foreach (var tableGroup in tableSchemas.GroupBy(t => t.TableName))
            {
                var tableName = tableGroup.Key;
                var columnDefinitions = BuildColumnDefinitions(tableGroup);
                AppendTableDefinition(schemaScript, tableName, columnDefinitions);
            }
        }
    }

    private static async Task<IEnumerable<TableSchemaInfo>> GetBatchTableColumnDefinitions(SqlConnection connection, List<string> tableNames)
    {
        var query = @"
            SELECT 
                c.TABLE_NAME as TableName,
                c.COLUMN_NAME as ColumnName,
                c.DATA_TYPE as DataType,
                c.CHARACTER_MAXIMUM_LENGTH as CharacterMaxLength,
                c.NUMERIC_PRECISION as NumericPrecision,
                c.NUMERIC_SCALE as NumericScale,
                CASE WHEN c.IS_NULLABLE = 'YES' THEN 1 ELSE 0 END as IsNullable,
                ISNULL(COLUMNPROPERTY(OBJECT_ID(c.TABLE_SCHEMA + '.' + c.TABLE_NAME), c.COLUMN_NAME, 'IsIdentity'), 0) as IsIdentity
            FROM INFORMATION_SCHEMA.COLUMNS c
            WHERE c.TABLE_NAME IN @TableNames
            ORDER BY c.TABLE_NAME, c.ORDINAL_POSITION";

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

        // 添加資料類型的長度/精度/小數點
        if (column.CharacterMaxLength.HasValue)
        {
            sb.Append(column.CharacterMaxLength == -1 ? "(MAX)" : $"({column.CharacterMaxLength})");
        }
        else if (column.NumericPrecision.HasValue)
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
                t.name as ObjectName,
                'U' as ObjectType,
                OBJECT_DEFINITION(t.object_id) as Definition
            FROM sys.tables t
            WHERE t.is_ms_shipped = 0";

        return await connection.QueryAsync<DatabaseInfo>(tableQuery);
    }

    private static async Task SaveSchemaScript(string schemaScript, string targetPath)
    {
        string filePath = Path.Combine(targetPath, "CreateDatabase.sql");
        await File.WriteAllTextAsync(filePath, schemaScript);
        Console.WriteLine($"資料庫結構已成功導出到 {filePath}");
    }
}
