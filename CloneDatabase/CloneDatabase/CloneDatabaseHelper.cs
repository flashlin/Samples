using Microsoft.Extensions.Options;

namespace CloneDatabase;

public class CloneDatabaseHelper
{
    private readonly DbConfig _dbConfig;

    public CloneDatabaseHelper(IOptions<DbConfig> dbConfig)
    {
        _dbConfig = dbConfig.Value;
    }
    
    public void Clone()
    {
        var sourceDb = new SqlDb(_dbConfig.SourceServer);
        foreach (var dbInfo in sourceDb.QueryDatabases())
        {
            if (dbInfo.Name.Equals("tempdb", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            Console.WriteLine($"{dbInfo.Name} {dbInfo.CreateDate:yyyy-MM-dd}");

            var tables = sourceDb.QueryTables(dbInfo.Name);
            foreach (var tableInfo in tables)
            {
                Console.WriteLine($"\t{tableInfo.Name}");
                
                var fields = sourceDb.QueryFields(dbInfo.Name, tableInfo.Name);
                foreach (var field in fields)
                {
                    Console.WriteLine($"\t\t{field.Name} {field.IsNullable}");
                }
            }
        }
    }
}

public class TableFieldInfo
{
    public string Name { get; set; }
    public string DataType { get; set; }
    public int MaxLength { get; set; }
    public int Precision { get; set; }
    public int Scale { get; set; }
    public bool IsNullable { get; set; }
    public bool IsPrimaryKey { get; set; }
}