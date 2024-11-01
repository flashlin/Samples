using Microsoft.EntityFrameworkCore;

namespace SqlSharpLit;

public class DynamicDbContext : DbContext
{
    public DynamicDbContext(DbContextOptions<DynamicDbContext>? options)
        : base(options ?? CreateDbContextOptions(null))
    {
    }
    
    public static DbContextOptions<DynamicDbContext> CreateInMemoryDbContextOptions()
    {
        return new DbContextOptionsBuilder<DynamicDbContext>()
            .UseInMemoryDatabase("InMemoryDb")
            .Options;
    }

    public static DbContextOptions<DynamicDbContext> CreateDbContextOptions(string? connectionString)
    {
        connectionString ??= @".\\SQLExpress;Integrated Security=true;";
        return new DbContextOptionsBuilder<DynamicDbContext>()
            .UseSqlServer(connectionString)
            .Options;
    }

    public List<Dictionary<string, string>> ExportTableData(string tableName)
    {
        var fields = GetTableSchema(tableName);
        string? accumulator = null;
        var key = fields.First(x => x.IsPk);
        var result = new List<Dictionary<string, string>>(); 
        do
        {
            var data = GetTopNTableData(1000, tableName, fields, accumulator);
            if (data.Count == 0)
            {
                break;
            }
            result.AddRange(data);
            accumulator = data.Last()[key.Name];
        } while (true);
        return result;
    }

    public List<Dictionary<string, string>> GetTopNTableData(int topCount, string tableName,
        List<TableSchemaEntity> fields,
        string? accumulator)
    {
        var fieldNames = string.Join(",", fields.Select(x => x.Name));
        var key = fields.First(x => x.IsPk);
        var idKeyName = key.Name;
        string sql;
        if (IsStringType(key)) 
        {
            accumulator ??= string.Empty;
            sql = $@"SELECT TOP {topCount} {fieldNames} FROM {tableName} WHERE {idKeyName} > '{accumulator}'";
        }
        else
        {
            accumulator ??= "0";
            sql = $@"SELECT TOP {topCount} {fieldNames} FROM {tableName} WHERE {idKeyName} > {accumulator}";
        }
        return Database.SqlQueryRaw<Dictionary<string, string>>(sql).ToList();
    }

    private static bool IsStringType(TableSchemaEntity key)
    {
        return key.Type.ToLower().StartsWith("varchar") || key.Type.ToLower().StartsWith("nvarchar");
    }

    public List<TableSchemaEntity> GetTableSchema(string tableName)
    {
        return Database.SqlQueryRaw<TableSchemaEntity>(GetTableSchemaSql(tableName))
            .ToList();
    }

    private string GetTableSchemaSql(string tableName)
    {
        return $@"""
SELECT 
    c.COLUMN_NAME AS [Name],
    CASE 
        WHEN c.DATA_TYPE IN ('char', 'varchar', 'nchar', 'nvarchar') 
            THEN c.DATA_TYPE + '(' + 
                CASE 
                    WHEN c.CHARACTER_MAXIMUM_LENGTH = -1 THEN 'MAX'
                    ELSE CAST(c.CHARACTER_MAXIMUM_LENGTH AS VARCHAR)
                END + ')'
        WHEN c.DATA_TYPE IN ('decimal', 'numeric') 
            THEN c.DATA_TYPE + '(' + 
                CAST(c.NUMERIC_PRECISION AS VARCHAR) + ',' + 
                CAST(c.NUMERIC_SCALE AS VARCHAR) + ')'
        ELSE c.DATA_TYPE
    END AS [Type],
    CASE 
        WHEN c.IS_NULLABLE = 'YES' THEN 'NULL'
        ELSE 'NOT NULL'
    END AS [IsNull],
    CASE 
        WHEN pk.COLUMN_NAME IS NOT NULL THEN 'YES'
        ELSE 'NO'
    END AS [IsPK]
FROM 
    INFORMATION_SCHEMA.COLUMNS c
LEFT JOIN 
    (SELECT COLUMN_NAME
     FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
     JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu 
         ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
     WHERE tc.TABLE_NAME = '{tableName}' AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
    ) pk ON c.COLUMN_NAME = pk.COLUMN_NAME
WHERE 
    c.TABLE_NAME = '{tableName}'
ORDER BY 
    c.ORDINAL_POSITION;
""";
    }
}

public class TableSchemaEntity
{
    public string Name { get; set; } = string.Empty;
    public string Type { get; set; } = string.Empty;
    public bool IsNull { get; set; }
    public bool IsPk { get; set; }
}