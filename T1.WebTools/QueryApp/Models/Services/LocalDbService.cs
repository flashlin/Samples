using Microsoft.EntityFrameworkCore;
using T1.SqlLocalData;

namespace QueryApp.Models.Services;

public interface ILocalDbService
{
    string GetDbConnectionString();
}

public class LocalDbService : ILocalDbService
{
    private readonly string _databaseName = "ReportDb";
    private readonly string _instanceName = "local_report_instance";
    private readonly SqlLocalDb _localDb;

    public LocalDbService(ILocalEnvironment localEnvironment)
    {
        _localDb = new SqlLocalDb(localEnvironment.AppLocation);
        InitializeLocalDb();
    }

    public string GetDbConnectionString()
    {
        return _localDb.GetDatabaseConnectionString(_instanceName, _databaseName);
    }

    public void QueryAllTables()
    {
    }

    private void InitializeLocalDb()
    {
        _localDb.EnsureInstanceCreated(_instanceName);
        _localDb.CreateDatabase(_instanceName, _databaseName);
    }
}

public class ReportDbContext : DbContext
{
    private readonly string _connectionString;

    public ReportDbContext(ILocalDbService localDbService)
    {
        _connectionString = localDbService.GetDbConnectionString();
    }
    
    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        optionsBuilder.UseSqlServer(_connectionString);
    }
}