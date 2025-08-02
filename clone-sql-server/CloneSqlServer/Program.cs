using CloneSqlServer;
using CloneSqlServer.Kit;

public class SqlBoxerEnv
{
    public string SqlSaPassword { get; set; } = string.Empty;

    public static SqlBoxerEnv LoadFromEnvironment()
    {
        var password = Environment.GetEnvironmentVariable("SQL_SA_PASSWORD");
        if (string.IsNullOrEmpty(password))
        {
            throw new InvalidOperationException("環境變數 SQL_SA_PASSWORD 未設定，請設定 SQL Server SA 密碼");
        }

        return new SqlBoxerEnv
        {
            SqlSaPassword = password
        };
    }
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

        var env = SqlBoxerEnv.LoadFromEnvironment();
        string connectionString = BuildConnectionString(args[0], env);
        string targetPath = args[1];

        // 確保目標目錄存在
        Directory.CreateDirectory(targetPath);

        Console.WriteLine($"連接字串: {connectionString}");
        await using var db = new SqlDbContext();
        await db.OpenAsync(connectionString);
        Console.WriteLine("成功連接到 SQL Server");
    }

    private static string BuildConnectionString(string server, SqlBoxerEnv env)
    {
        var serverParts = server.Split(':');
        var serverName = serverParts[0];
        var port = serverParts.Length > 1 ? $",{serverParts[1]}" : string.Empty;
        return $"Server={serverName}{port};User ID=sa;Password={env.SqlSaPassword};TrustServerCertificate=True;";
    }
}
