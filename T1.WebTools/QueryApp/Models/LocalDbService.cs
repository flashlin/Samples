using QueryKits.Services;
using T1.SqlLocalData;

namespace QueryApp.Models;

public class LocalDbService : ILocalDbService
{
    private readonly string _instanceName = "local_report_instance";
    private readonly SqlLocalDb _localDb;
    private readonly ILocalEnvironment _localEnvironment;
    public const string DatabaseName = "QueryDb";

    public LocalDbService(ILocalEnvironment localEnvironment)
    {
        _localEnvironment = localEnvironment;
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
        var dbFile = Path.Combine(_localEnvironment.AppLocation, $"{DatabaseName}.mdf");
        if (!File.Exists(dbFile))
        {
            if (_localDb.IsDatabaseExists(_instanceName, DatabaseName))
            {
                _localDb.DropDatabase(_instanceName, DatabaseName);
            }
            _localDb.CreateDatabase(_instanceName, DatabaseName);
        }
    }
}