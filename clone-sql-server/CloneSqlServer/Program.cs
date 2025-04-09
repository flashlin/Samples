using Microsoft.Data.SqlClient;
using System.Text;

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
                List<string> databases = new List<string>();
                using (SqlCommand command = new SqlCommand(
                    "SELECT name FROM sys.databases WHERE database_id > 4", connection))
                {
                    using (SqlDataReader reader = await command.ExecuteReaderAsync())
                    {
                        while (await reader.ReadAsync())
                        {
                            databases.Add(reader.GetString(0));
                        }
                    }
                }

                StringBuilder schemaScript = new StringBuilder();

                foreach (string database in databases)
                {
                    // 切換資料庫
                    connection.ChangeDatabase(database);
                    
                    // 獲取資料庫結構
                    using (SqlCommand command = new SqlCommand(@"
                        SELECT OBJECT_DEFINITION(object_id) as Definition
                        FROM sys.objects
                        WHERE type in ('U', 'P', 'V', 'TR', 'FN')
                        AND is_ms_shipped = 0", connection))
                    {
                        schemaScript.AppendLine($"USE [{database}]");
                        schemaScript.AppendLine("GO");
                        
                        using (SqlDataReader reader = await command.ExecuteReaderAsync())
                        {
                            while (await reader.ReadAsync())
                            {
                                if (!reader.IsDBNull(0))
                                {
                                    schemaScript.AppendLine(reader.GetString(0));
                                    schemaScript.AppendLine("GO");
                                }
                            }
                        }
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
