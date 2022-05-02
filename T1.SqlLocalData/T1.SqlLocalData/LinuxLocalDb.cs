using System.Collections.Generic;
using Dapper;
using Microsoft.Data.SqlClient;

namespace T1.SqlLocalData;

public class LinuxLocalDb : ISqlLocalDb
{
    private readonly string _connectionString;

    public LinuxLocalDb(string connectionString = null)
    {
        if (string.IsNullOrEmpty(connectionString))
        {
            connectionString = "Server=localhost;User=sa;Password=1Secure*Password1;";
        }

        _connectionString = connectionString;
    }

	public void CreateDatabase(string instanceName, string databaseName)
	{
		var sql = $@"
IF NOT EXISTS(SELECT * FROM sys.databases WHERE name = '{databaseName}')
BEGIN
	CREATE DATABASE {databaseName}
END";
		ExecuteNonQueryRawSql(sql);
	}

	public void DeleteDatabaseFile(string databaseName)
	{
	}

	public void EnsureInstanceCreated(string instanceName)
	{
	}

	public void ExecuteNonQueryRawSql(string instanceName, string sql, object parameter = null)
	{
		ExecuteNonQueryRawSql(sql);
	}

	public void ForceDropDatabase(string instanceName, string databaseName)
	{
		var sql = @$"DROP DATABASE IF EXISTS {databaseName}";
		ExecuteNonQueryRawSql(sql);
	}

	public string GetDatabaseConnectionString(string instanceName, string databaseName)
	{
		return $"Server=localhost;User=sa;Password=1Secure*Password1;Database={databaseName}";
	}

	public void KillAllConnections(string instanceName, string databaseName)
	{
		var sql = $@"DECLARE @DatabaseName nvarchar(50)=N'{databaseName}'
DECLARE @SQL varchar(max)
SELECT @SQL = COALESCE(@SQL,'') + 'Kill ' + Convert(varchar, SPId) + ';'
FROM MASTER..SysProcesses
WHERE DBId = DB_ID(@DatabaseName) AND SPId <> @@SPId
EXEC(@SQL)";
		ExecuteNonQueryRawSql(sql);
	}

	public IEnumerable<T> QuerySqlRaw<T>(string instanceName, string sql, object parameter = null)
	{
		return QuerySqlRaw<T>(sql, parameter);
	}

	private void ExecuteNonQueryRawSql(string sql)
	{
        using var conn = new SqlConnection(_connectionString);
        conn.Execute(sql);
    }


    private IEnumerable<T> QuerySqlRaw<T>(string sql, object parameter = null)
    {
        using var conn = new SqlConnection(_connectionString);
        return conn.Query<T>(sql, parameter);
    }
}