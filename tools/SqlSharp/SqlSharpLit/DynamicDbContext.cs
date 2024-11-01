using Microsoft.EntityFrameworkCore;

namespace SqlSharpLit;

public class DynamicDbContext : DbContext
{
    public DynamicDbContext(DbContextOptions<DynamicDbContext>? options) 
        : base(options ?? CreateDbContextOptions(null)) 
    {
    }

    public static DbContextOptions<DynamicDbContext> CreateDbContextOptions(string? connectionString)
    {
        connectionString ??= @".\\SQLExpress;Integrated Security=true;";
        var options = new DbContextOptionsBuilder<DynamicDbContext>()
            .UseSqlServer(connectionString)
            .Options;
        return options;
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