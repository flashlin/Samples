using T1.SqlLocalData;

namespace QueryApp.Models.Services;

public class LocalDbService : ILocalDbService
{
    private readonly string _instanceName = "local_report_instance";
    private readonly SqlLocalDb _localDb;
    public const string DatabaseName = "ReportDb";

    public LocalDbService(ILocalEnvironment localEnvironment)
    {
        _localDb = new SqlLocalDb(localEnvironment.AppLocation);
        InitializeLocalDb();
    }

    public string GetDbConnectionString()
    {
        return _localDb.GetDatabaseConnectionString(_instanceName, DatabaseName);
    }

    private void InitializeLocalDb()
    {
        _localDb.EnsureInstanceCreated(_instanceName);
        _localDb.CreateDatabase(_instanceName, DatabaseName);
    }
}