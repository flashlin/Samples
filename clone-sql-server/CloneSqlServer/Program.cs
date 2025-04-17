using Microsoft.Data.SqlClient;
using System.Text;
using CloneSqlServer;
using Dapper;

public class SqlBoxerEnv
{
    public string SqlSaPassword { get; set; }

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
        await using var connection = new SqlConnection(connectionString);
        await connection.OpenAsync();
        Console.WriteLine("成功連接到 SQL Server");

        var context = await GenerateContext.Initialize(connection);
        await GenerateDatabaseSchemaScripts(connection, context, targetPath);
    }

    private static string BuildConnectionString(string server, SqlBoxerEnv env)
    {
        string[] serverParts = server.Split(':');
        string serverName = serverParts[0];
        string port = serverParts.Length > 1 ? $",{serverParts[1]}" : string.Empty;
        return $"Server={serverName}{port};User ID=sa;Password={env.SqlSaPassword};TrustServerCertificate=True;";
    }

    private static async Task GenerateDatabaseSchemaScripts(SqlConnection connection, GenerateContext context, string targetPath)
    {
        var combinedScript = new StringBuilder();
        combinedScript.AppendLine("-- Combined Database Creation Script");
        combinedScript.AppendLine($"-- Generated at {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        combinedScript.AppendLine();

        foreach (string database in context.Databases)
        {
            Console.WriteLine($"開始產生資料庫 {database} 的結構指令碼...");
            var schemaScript = await GenerateDatabaseSchemaScript(connection, context, database);
            
            // 儲存個別資料庫腳本
            await SaveDatabaseSchemaScript(schemaScript, targetPath, database);
            Console.WriteLine($"完成產生資料庫 {database} 的結構指令碼");

            // 加入到整合腳本
            combinedScript.AppendLine($"-- Start Database: {database}");
            combinedScript.AppendLine(schemaScript);
            combinedScript.AppendLine($"-- End Database: {database}");
            combinedScript.AppendLine();
        }

        // 儲存整合腳本
        string combinedFilePath = Path.Combine(targetPath, "CreateDatabase.sql");
        await File.WriteAllTextAsync(combinedFilePath, combinedScript.ToString());
        Console.WriteLine($"完成產生整合資料庫腳本：{combinedFilePath}");
    }

    private static async Task<string> GenerateDatabaseSchemaScript(SqlConnection connection, GenerateContext context, string database)
    {
        var schemaScript = new StringBuilder();
        await GenerateDatabaseObjects(connection, database, schemaScript, context);
        return schemaScript.ToString();
    }

    private static async Task SaveDatabaseSchemaScript(string schemaScript, string targetPath, string database)
    {
        string filePath = Path.Combine(targetPath, $"{database}_Schema.sql");
        await File.WriteAllTextAsync(filePath, schemaScript);
        Console.WriteLine($"資料庫 {database} 的結構已成功導出到 {filePath}");
    }

    private static async Task GenerateDatabaseObjects(SqlConnection connection, string database, StringBuilder schemaScript, GenerateContext context)
    {
        AppendCreateDatabaseScript(schemaScript, database);

        connection.ChangeDatabase(database);
        
        schemaScript.AppendLine($"USE [{database}]");
        schemaScript.AppendLine("GO");

        Console.WriteLine($"Creating database objects for {database}");
        await GenerateTableDefinitions(schemaScript, context, database);
        GenerateTableIndexObjects(schemaScript, context, database);
        GenerateTableConstraints(schemaScript, context, database);
        await GenerateUserFunctions(connection, schemaScript);
        await GenerateUserDefineTypes(connection, schemaScript, context, database);
        await GenerateViews(connection, schemaScript);
        await GenerateStoredProcedures(connection, schemaScript, context, database);
        GenerateLoginUsers(schemaScript, context, database);
        GenerateRolePermissions(schemaScript, context);
        await GenerateUserDefineTypesRolePermission(schemaScript, context, database);
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

    private static async Task GenerateUserDefineTypes(SqlConnection connection, StringBuilder schemaScript, GenerateContext context, string database)
    {
        var userDefinedTypes = context.UserDefinedTypes
            .Where(udt => udt.Name == database)
            .OrderBy(udt => udt.ObjectName);

        foreach (var udt in userDefinedTypes)
        {
            if (!string.IsNullOrEmpty(udt.Definition))
            {
                schemaScript.AppendLine($"-- User-Defined Type: {udt.ObjectName}");
                schemaScript.AppendLine($"IF TYPE_ID(N'{udt.ObjectName}') IS NOT NULL");
                schemaScript.AppendLine($"    DROP TYPE [{udt.ObjectName}]");
                schemaScript.AppendLine("GO");
                schemaScript.AppendLine(udt.Definition);
                schemaScript.AppendLine("GO");
                schemaScript.AppendLine();
            }
        }
    }

    private static Task GenerateUserDefineTypesRolePermission(StringBuilder schemaScript, GenerateContext context, string database)
    {
        var udtWithRoles = context.UserDefinedTypeWithRoles
            .Where(udt => udt.DatabaseName == database)
            .ToList();

        if (udtWithRoles.Any())
        {
            schemaScript.AppendLine($"-- User-Defined Type Role Permissions for {database}");
            foreach (var udt in udtWithRoles)
            {
                if (!string.IsNullOrEmpty(udt.RoleNames))
                {
                    foreach (var role in udt.RoleNamesList)
                    {
                        schemaScript.AppendLine($"GRANT EXECUTE ON TYPE::[{udt.TypeName}] TO [{role}]");
                        schemaScript.AppendLine("GO");
                    }
                }
            }
            schemaScript.AppendLine();
        }

        return Task.CompletedTask;
    }

    private static async Task GenerateStoredProcedures(SqlConnection connection, StringBuilder schemaScript, GenerateContext context, string database)
    {
        schemaScript.AppendLine($"USE [{database}]");
        schemaScript.AppendLine("GO");
        schemaScript.AppendLine();

        schemaScript.AppendLine($"-- AllStoreProceduresCount={context.StoreProcedures.Count}");
        schemaScript.AppendLine($"-- {database} AllStoreProceduresCount={context.StoreProcedures.Count(x => x.DatabaseName==database)}");

        await GenerateIndependentStoredProcedures(schemaScript, context, database);
        await GenerateDependentStoredProcedures(schemaScript, context, database);
    }

    /// <summary>
    /// Generate independent stored procedures (those that don't depend on other stored procedures)
    /// </summary>
    private static Task GenerateIndependentStoredProcedures(StringBuilder schemaScript, GenerateContext context, string database)
    {
        Console.WriteLine($"Generating independent stored procedures for database {database}...");
        var databaseSps = context.IndependentStoreProcedures.Where(sp => sp.DatabaseName == database);
        
        foreach (var sp in databaseSps)
        {
            if (!string.IsNullOrEmpty(sp.Definition))
            {
                AppendStoredProcedureDefinition(schemaScript, sp, isIndependent: true);
            }
        }
        return Task.CompletedTask;
    }

    /// <summary>
    /// Generate dependent stored procedures (those that depend on other stored procedures)
    /// </summary>
    private static Task GenerateDependentStoredProcedures(StringBuilder schemaScript, GenerateContext context, string database)
    {
        Console.WriteLine($"Generating dependent stored procedures for database {database}...");
        var independentSpNames = GetIndependentStoredProcedureNames(context, database);
        var dependentSps = GetDependentStoredProcedures(context, independentSpNames, database);

        foreach (var sp in dependentSps)
        {
            if (!string.IsNullOrEmpty(sp.Definition))
            {
                AppendStoredProcedureDefinition(schemaScript, sp, isIndependent: false);
            }
        }
        return Task.CompletedTask;
    }

    /// <summary>
    /// 取得獨立預存程序的完整名稱清單
    /// </summary>
    private static HashSet<string> GetIndependentStoredProcedureNames(GenerateContext context, string database)
    {
        return context.IndependentStoreProcedures
            .Where(sp => sp.DatabaseName == database)
            .Select(sp => $"{sp.DatabaseName}.{sp.StoreProcedureName}")
            .ToHashSet();
    }

    /// <summary>
    /// 取得相依的預存程序清單
    /// </summary>
    private static IEnumerable<StoreProcedureInfo> GetDependentStoredProcedures(
        GenerateContext context, 
        HashSet<string> independentSpNames,
        string database)
    {
        return context.StoreProcedures
            .Where(sp => sp.DatabaseName == database && 
                        !independentSpNames.Contains($"{sp.DatabaseName}.{sp.StoreProcedureName}"));
    }

    /// <summary>
    /// 將預存程序定義加入到腳本中
    /// </summary>
    private static void AppendStoredProcedureDefinition(
        StringBuilder schemaScript, 
        StoreProcedureInfo sp,
        bool isIndependent)
    {
        var text = new StringBuilder();
        var spType = isIndependent ? "Independent" : "Dependent";
        text.AppendLine($"-- {spType} Stored Procedure: [{sp.DatabaseName}].[{sp.StoreProcedureName}]");
        text.AppendLine(sp.Definition);
        text.AppendLine("GO");
        text.AppendLine();
        schemaScript.AppendLine(text.ToString());
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

    private static Task GenerateTableDefinitions(StringBuilder schemaScript, GenerateContext context, string database)
    {
        var tables = context.Tables[database];
        var tableSchemas = context.TableSchemas[database];

        foreach (var table in tables)
        {
            var tableName = table.ObjectName!;
            var columnDefinitions = BuildColumnDefinitions(tableSchemas.Where(t => t.TableName == tableName)
                .GroupBy(t => t.TableName).First());

            Console.WriteLine($"Processing table {tableName}");
            AppendTableDefinition(schemaScript, tableName, columnDefinitions);
        }
        return Task.CompletedTask;
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

    private static void GenerateTableIndexObjects(StringBuilder schemaScript, GenerateContext context, string database)
    {
        var tableIndexes = context.TableIndexes[database];
        
        GeneratePrimaryKeys(schemaScript, tableIndexes);
        GenerateForeignKeys(schemaScript, tableIndexes);
        GenerateIndexes(schemaScript, tableIndexes);
    }

    private static void GeneratePrimaryKeys(StringBuilder schemaScript, List<TableIndexSchema> tableIndexes)
    {
        var primaryKeys = tableIndexes.Where(i => i.IsPrimaryKey).ToList();
        if (!primaryKeys.Any())
        {
            return;
        }

        foreach (var pk in primaryKeys)
        {
            var text = new StringBuilder();
            text.AppendLine($"-- Primary Key: {pk.IndexName} on {pk.TableName}");
            text.AppendLine($"ALTER TABLE [{pk.TableName}] ADD CONSTRAINT [{pk.IndexName}]");
            text.AppendLine($"    PRIMARY KEY {(pk.IsClustered ? "CLUSTERED" : "NONCLUSTERED")} (");
            text.AppendLine($"        {string.Join(",\n        ", pk.Columns.Select(c => $"[{c}]"))}");
            text.AppendLine("    )");
            text.AppendLine("GO");
            text.AppendLine();
            schemaScript.AppendLine(text.ToString());
        }
    }

    private static void GenerateForeignKeys(StringBuilder schemaScript, List<TableIndexSchema> tableIndexes)
    {
        var foreignKeys = tableIndexes.Where(i => i.IndexType == "FK").ToList();
        if (!foreignKeys.Any())
        {
            return;
        }

        foreach (var fk in foreignKeys)
        {
            var text = new StringBuilder();
            text.AppendLine($"-- Foreign Key: {fk.IndexName} on {fk.TableName}");
            text.AppendLine($"ALTER TABLE [{fk.TableName}] ADD CONSTRAINT [{fk.IndexName}]");
            text.AppendLine($"    FOREIGN KEY (");
            text.AppendLine($"        {string.Join(",\n        ", fk.Columns.Select(c => $"[{c}]"))}");
            text.AppendLine("    )");
            text.AppendLine($"    REFERENCES [{fk.ReferencedTableName}] (");
            text.AppendLine($"        {string.Join(",\n        ", fk.ReferencedColumns.Select(c => $"[{c}]"))}");
            text.AppendLine("    )");
            text.AppendLine("GO");
            text.AppendLine();
            schemaScript.AppendLine(text.ToString());
        }
    }

    private static void GenerateIndexes(StringBuilder schemaScript, List<TableIndexSchema> tableIndexes)
    {
        var indexes = tableIndexes.Where(i => i.IndexType == "INDEX").ToList();
        if (!indexes.Any())
        {
            return;
        }

        foreach (var idx in indexes)
        {
            var text = new StringBuilder();
            text.AppendLine($"-- Index: {idx.IndexName} on {idx.TableName}");
            text.AppendLine($"CREATE {(idx.IsUnique ? "UNIQUE " : "")}{(idx.IsClustered ? "CLUSTERED" : "NONCLUSTERED")} INDEX [{idx.IndexName}]");
            text.AppendLine($"    ON [{idx.TableName}] (");
            text.AppendLine($"        {string.Join(",\n        ", idx.Columns.Select(c => $"[{c}]"))}");
            text.AppendLine("    )");
            text.AppendLine("GO");
            text.AppendLine();
            schemaScript.AppendLine(text.ToString());
        }
    }

    private static string GetPasswordFromEnv()
    {
        var envPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ".env");
        if (!File.Exists(envPath))
        {
            throw new Exception($"警告：{envPath} 找不到 .env 檔案，請設定密碼");
        }

        var envContent = File.ReadAllLines(envPath);
        var passwordLine = envContent.FirstOrDefault(line => line.StartsWith("PASSWORD="));
        return passwordLine.Split('=')[1].Trim();
    }

    private static void GenerateLoginUsers(StringBuilder schemaScript, GenerateContext context, string database)
    {
        GenerateCreateLogins(schemaScript, context.LoginNames.Select(l => l.LoginName).ToList(), database);
        GenerateCreateDatabaseRoles(schemaScript, context, database);
        GenerateAddRoleMembers(schemaScript, context.LoginRoles, database);
    }

    private static void GenerateCreateDatabaseRoles(StringBuilder schemaScript, GenerateContext context,
        string database)
    {
        var dbRoles = context.DatabaseRoleNames
            .Where(x=>x.DatabaseName == database)
            .ToList();

        schemaScript.AppendLine($"-- Create database roles for {database}");
        schemaScript.AppendLine($"USE [{database}]");
        schemaScript.AppendLine("GO");

        foreach (var role in dbRoles)
        {
            schemaScript.AppendLine($@"
IF NOT EXISTS (SELECT name FROM sys.database_principals WHERE name = '{role.RoleName}' AND type = 'R')
BEGIN
    CREATE ROLE [{role.RoleName}]
END
GO");
        }
        schemaScript.AppendLine();
    }

    private static void GenerateCreateLogins(StringBuilder schemaScript, List<string> loginNames, string database)
    {
        var passwordFromEnv = GetPasswordFromEnv();
        foreach (var loginName in loginNames)
        {
            schemaScript.AppendLine($@"
-- Create login {loginName}
IF NOT EXISTS (SELECT name FROM sys.server_principals WHERE name = '{loginName}')
BEGIN
    CREATE LOGIN [{loginName}] WITH PASSWORD = '{passwordFromEnv}'
END
GO

-- Create database users for the login
");
            schemaScript.AppendLine($@"
USE [{database}]
GO
IF NOT EXISTS (SELECT name FROM sys.database_principals WHERE name = '{loginName}')
BEGIN
    CREATE USER [{loginName}] FOR LOGIN [{loginName}]
END
GO");
            schemaScript.AppendLine();
        }
    }

    private static void GenerateAddRoleMembers(StringBuilder schemaScript, List<LoginRoleInfo> loginRoles,
        string database)
    {
        var dbRoles = loginRoles
            .Where(x=>x.DatabaseName == database)
            .ToList();

        schemaScript.AppendLine($"-- Add role members for {database}");
        schemaScript.AppendLine($"USE [{database}]");
        schemaScript.AppendLine("GO");

        foreach (var role in dbRoles)
        {
            schemaScript.AppendLine($"ALTER ROLE [{role.RoleName}] ADD MEMBER [{role.LoginName}]");
            schemaScript.AppendLine("GO");
        }
        schemaScript.AppendLine();
    }

    private static void GenerateRolePermissions(StringBuilder schemaScript, GenerateContext context)
    {
        var permissions = context.StoreProcedurePermissions
            .GroupBy(p => p.DatabaseName)
            .OrderBy(g => g.Key);

        foreach (var dbPermissions in permissions)
        {
            schemaScript.AppendLine($"-- Permissions for database {dbPermissions.Key}");
            schemaScript.AppendLine($"USE [{dbPermissions.Key}]");
            schemaScript.AppendLine("GO");

            foreach (var permission in dbPermissions)
            {
                schemaScript.AppendLine($"GRANT {permission.PermissionName} ON [{permission.StoreProcedureName}] TO [{permission.RoleName}]");
                schemaScript.AppendLine("GO");
            }
            schemaScript.AppendLine();
        }
    }

    private static void GenerateTableConstraints(StringBuilder schemaScript, GenerateContext context, string database)
    {
        var constraints = context.TableConstraints
            .Where(c => c.DatabaseName == database)
            .GroupBy(c => c.TableName)
            .OrderBy(g => g.Key);

        foreach (var tableGroup in constraints)
        {
            schemaScript.AppendLine($"-- Constraints for Table: {tableGroup.Key}");
            foreach (var constraint in tableGroup.OrderBy(c => c.ConstraintName))
            {
                if (string.IsNullOrEmpty(constraint.ColumnName))
                {
                    // Check constraint
                    schemaScript.AppendLine($"ALTER TABLE [{tableGroup.Key}] ADD CONSTRAINT [{constraint.ConstraintName}] CHECK {constraint.ConstraintDefine}");
                }
                else
                {
                    // Default constraint
                    schemaScript.AppendLine($"ALTER TABLE [{tableGroup.Key}] ADD CONSTRAINT [{constraint.ConstraintName}] DEFAULT {constraint.ConstraintDefine} FOR [{constraint.ColumnName}]");
                }
                schemaScript.AppendLine("GO");
                schemaScript.AppendLine();
            }
        }
    }
}
