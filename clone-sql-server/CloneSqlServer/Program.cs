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

        string server = args[0];
        string[] serverParts = server.Split(':');
        string connectionString = $"Server={serverParts[0]},{serverParts[1]};Integrated Security=True;TrustServerCertificate=True;";

        try
        {
            using (SqlConnection connection = new SqlConnection(connectionString))
            {
                await connection.OpenAsync();
                Console.WriteLine("成功連接到 SQL Server");

                // 獲取所有資料庫
                var databases = await connection.QueryAsync<string>(
                    "SELECT name FROM sys.databases WHERE database_id > 4"
                );

                StringBuilder schemaScript = new StringBuilder();

                foreach (string database in databases)
                {
                    // 切換資料庫
                    connection.ChangeDatabase(database);
                    
                    // 使用 Dapper 獲取資料庫物件定義
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

                    schemaScript.AppendLine($"USE [{database}]");
                    schemaScript.AppendLine("GO");

                    foreach (var obj in dbObjects)
                    {
                        if (!string.IsNullOrEmpty(obj.Definition))
                        {
                            schemaScript.AppendLine($"-- Object: {obj.ObjectName} ({obj.ObjectType})");
                            schemaScript.AppendLine(obj.Definition);
                            schemaScript.AppendLine("GO");
                        }
                    }

                    // 獲取資料表結構
                    var tableQuery = @"
                        SELECT 
                            DB_NAME() as Name,
                            t.name as ObjectName,
                            'U' as ObjectType,
                            OBJECT_DEFINITION(t.object_id) as Definition
                        FROM sys.tables t
                        WHERE t.is_ms_shipped = 0";

                    var tables = await connection.QueryAsync<DatabaseInfo>(tableQuery);

                    foreach (var table in tables)
                    {
                        schemaScript.AppendLine($"-- Table: {table.ObjectName}");
                        var tableDefinition = await connection.QueryFirstAsync<string>(@"
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
                            new { TableName = table.ObjectName }
                        );

                        schemaScript.AppendLine($"CREATE TABLE [{table.ObjectName}] (");
                        schemaScript.AppendLine($"    {tableDefinition}");
                        schemaScript.AppendLine(")");
                        schemaScript.AppendLine("GO");
                    }
                }

                // 寫入檔案
                await File.WriteAllTextAsync("CreateDatabase.sql", schemaScript.ToString());
                Console.WriteLine("資料庫結構已成功導出到 CreateDatabase.sql");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"錯誤: {ex.Message}");
        }
    }
}
