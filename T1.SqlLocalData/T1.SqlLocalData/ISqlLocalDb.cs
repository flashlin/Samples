using System.Collections.Generic;

namespace T1.SqlLocalData;

public interface ISqlLocalDb
{
    void CreateDatabase(string instanceName, string databaseName);
    void EnsureInstanceCreated(string instanceName);
    void ForceDropDatabase(string instanceName, string databaseName);
    void DeleteDatabaseFile(string databaseName);
    void ExecuteNonQueryRawSql(string instanceName, string sql, object parameter = null);
    IEnumerable<T> QuerySqlRaw<T>(string instanceName, string sql, object parameter = null);
    void KillAllConnections(string instanceName, string databaseName);
    string GetDatabaseConnectionString(string instanceName, string databaseName);
}