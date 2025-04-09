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

class Program
{
    static async Task Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("請提供 SQL Server 連接字串，例如: 127.0.0.1:3390");
            return;
        }

        string connectionString = BuildConnectionString(args[0]);
        using var connection = new SqlConnection(connectionString);
        await connection.OpenAsync();
        Console.WriteLine("成功連接到 SQL Server");

        var schemaScript = await GenerateDatabaseSchemaScript(connection);
        await SaveSchemaScript(schemaScript);
    }

    private static string BuildConnectionString(string server)
    {
        string[] serverParts = server.Split(':');
        return $"Server={serverParts[0]},{serverParts[1]};Integrated Security=True;TrustServerCertificate=True;";
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

        await GenerateStoredProceduresAndViews(connection, schemaScript);
        await GenerateTableDefinitions(connection, schemaScript);
    }

    private static void AppendCreateDatabaseScript(StringBuilder schemaScript, string database)
    {
        schemaScript.AppendLine($"IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'{database}')");
        schemaScript.AppendLine("BEGIN");
        schemaScript.AppendLine($"    CREATE DATABASE [{database}]");
        schemaScript.AppendLine("END");
        schemaScript.AppendLine("GO");
    }

    private static async Task GenerateStoredProceduresAndViews(SqlConnection connection, StringBuilder schemaScript)
    {
        var query = @"
            SELECT 
                DB_NAME() as Name,
                o.name as ObjectName,
                o.type as ObjectType,
                OBJECT_DEFINITION(o.object_id) as Definition
            FROM sys.objects o
            WHERE type in ('U', 'P', 'V', 'TR', 'FN')
            AND is_ms_shipped = 0";

        var dbObjects = await connection.QueryAsync<DatabaseInfo>(query);

        foreach (var obj in dbObjects)
        {
            if (!string.IsNullOrEmpty(obj.Definition))
            {
                schemaScript.AppendLine($"-- Object: {obj.ObjectName} ({obj.ObjectType})");
                schemaScript.AppendLine(obj.Definition);
                schemaScript.AppendLine("GO");
            }
        }
    }

    private static async Task GenerateTableDefinitions(SqlConnection connection, StringBuilder schemaScript)
    {
        var tables = await GetTables(connection);

        foreach (var table in tables)
        {
            var tableDefinition = await GetTableColumnDefinitions(connection, table.ObjectName);
            AppendTableDefinition(schemaScript, table.ObjectName, tableDefinition);
        }
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

    private static async Task<string> GetTableColumnDefinitions(SqlConnection connection, string? tableName)
    {
        return await connection.QueryFirstAsync<string>(@"
            SELECT 
                STRING_AGG(
                    CASE 
                        WHEN is_identity = 1 THEN COLUMN_NAME + ' ' + DATA_TYPE + 
                            CASE 
                                WHEN CHARACTER_MAXIMUM_LENGTH IS NOT NULL 
                                THEN '(' + CAST(CHARACTER_MAXIMUM_LENGTH AS VARCHAR) + ')' 
                                ELSE '' 
                            END + ' IDENTITY(1,1)'
                        ELSE COLUMN_NAME + ' ' + DATA_TYPE + 
                            CASE 
                                WHEN CHARACTER_MAXIMUM_LENGTH IS NOT NULL 
                                THEN '(' + CAST(CHARACTER_MAXIMUM_LENGTH AS VARCHAR) + ')' 
                                ELSE '' 
                            END
                    END + 
                    CASE 
                        WHEN IS_NULLABLE = 'NO' THEN ' NOT NULL'
                        ELSE ' NULL'
                    END,
                    ', '
                ) 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = @TableName",
            new { TableName = tableName }
        );
    }

    private static void AppendTableDefinition(StringBuilder schemaScript, string? tableName, string tableDefinition)
    {
        schemaScript.AppendLine($"-- Table: {tableName}");
        schemaScript.AppendLine($"CREATE TABLE [{tableName}] (");
        schemaScript.AppendLine($"    {tableDefinition}");
        schemaScript.AppendLine(")");
        schemaScript.AppendLine("GO");
    }

    private static async Task SaveSchemaScript(string schemaScript)
    {
        await File.WriteAllTextAsync("CreateDatabase.sql", schemaScript);
        Console.WriteLine("資料庫結構已成功導出到 CreateDatabase.sql");
    }
}
