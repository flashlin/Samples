using Microsoft.EntityFrameworkCore;

namespace QueryApp.Models.Services;

public class ReportDbContext : DbContext, IReportRepo
{
    private readonly string _connectionString;

    public ReportDbContext(ILocalDbService localDbService)
    {
        _connectionString = localDbService.GetDbConnectionString();
    }

    public List<string> GetAllTableNames()
    {
        var sql = $"""
            SELECT TABLE_NAME as TableName 
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_CATALOG='{LocalDbService.DatabaseName}'
            """;
        return Database.SqlQueryRaw<string>(sql)
            .ToList();
    }
    
    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        optionsBuilder.UseSqlServer(_connectionString);
    }
}