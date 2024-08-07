﻿using Dapper;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.IO;
using System.Linq;

namespace T1.SqlLocalData
{
	public class SqlLocalDb : ISqlLocalDb
	{
		private readonly string _dataFolder;
		string _installedLocation = @"C:\Program Files\Microsoft SQL Server\150\Tools\Binn";
		readonly string _sqllocaldbexe = @"SqlLocalDB.exe";

		public SqlLocalDb(string dataFolder)
		{
			_dataFolder = dataFolder;
		}

		public void SetInstalledLocation(string path)
		{
			_installedLocation = path;
		}

		public void CreateDatabase(string instanceName, string databaseName)
		{
			var mdfFile = Path.Combine(_dataFolder, $"{databaseName}.mdf");
			var logFile = Path.Combine(_dataFolder, $"{databaseName}.ldf");
			var sql = $"CREATE DATABASE {databaseName} ON PRIMARY " +
						 $"(NAME={databaseName}_Data, FILENAME='{mdfFile}', SIZE=2MB, MAXSIZE=10MB, FILEGROWTH= 10%) " +
						 $"LOG ON (NAME = {databaseName}_Log, FILENAME='{logFile}', SIZE=1MB, MAXSIZE=5MB, FILEGROWTH=10%) " +
						 "";

			ExecuteNonQueryRawSql(instanceName, sql);
		}

		public void CreateInstance(string instanceName)
		{
			ExecuteSqlLocalDbExe(@$"CREATE ""{instanceName}"" 15.0 -s");
		}

		public void DeleteDatabaseFile(string databaseName)
		{
			var mdfFile = Path.Combine(_dataFolder, $"{databaseName}.mdf");
			var logFile = Path.Combine(_dataFolder, $"{databaseName}.ldf");
			File.Delete(mdfFile);
			File.Delete(logFile);
		}

		public void DeleteInstance(string instanceName)
		{
			ExecuteSqlLocalDbExe(@$"delete ""{instanceName}""");
		}

		public void DetachDatabase(string instanceName, string databaseName)
		{
			ExecuteNonQueryRawSql(instanceName, $"EXEC sp_detach_db '{databaseName}', 'true';");
		}

		public void DropDatabase(string instanceName, string databaseName)
		{
			var sql = $"DROP DATABASE IF EXISTS {databaseName}";
			ExecuteNonQueryRawSql(instanceName, sql);
		}

		public void EnsureInstanceCreated(string instanceName)
		{
			if (IsInstanceExists(instanceName))
			{
				return;
			}

			CreateInstance(instanceName);
			StartInstance(instanceName);
		}

		public void ExecuteNonQueryRawSql(string instanceName, string sql, object parameter = null)
		{
			var connectionString = GetConnectionString(instanceName);
			using var myConn = new SqlConnection(connectionString);
			myConn.ExecuteScalar(sql, parameter);
		}

		public string ExecuteSqlLocalDbExe(string arguments)
		{
			var p = new ProcessHelper();
			var processFilename = $"{_installedLocation}\\{_sqllocaldbexe}";
			return p.Execute(processFilename, arguments);
		}

		public void ForceDropDatabase(string instanceName, string databaseName)
		{
			KillAllConnections(instanceName, databaseName);
			DropDatabase(instanceName, databaseName);
		}

		public void ForceStopInstance(string instanceName)
		{
			ExecuteSqlLocalDbExe(@$"stop ""{instanceName}"" -i -k");
		}

		public string GetDatabaseConnectionString(string instanceName, string databaseName)
		{
			var mdfFile = Path.Combine(_dataFolder, $"{databaseName}.mdf");
			return $"Server=(localdb)\\{instanceName};Integrated security=SSPI;AttachDbFileName={mdfFile};";
		}

		public IEnumerable<string> GetInstanceNames()
		{
			var result = ExecuteSqlLocalDbExe("i");
			var sr = new StringReader(result);
			do
			{
				var line = sr.ReadLine();
				if (line == null)
				{
					break;
				}

				yield return line;
			} while (true);
		}

		public bool IsDatabaseExists(string instanceName, string databaseName)
		{
			var sql = $@"if EXISTS(select dbid from (select DB_ID(@dbname) as dbid) t WHERE dbid IS NOT NULL)
select 1 
ELSE 
select 0";

			return QuerySqlRaw<bool>(instanceName, sql, new
			{
				dbname = databaseName
			}).First();
		}

		public bool IsInstanceExists(string instanceName)
		{
			return GetInstanceNames()
				 .Any(x => x == instanceName);
		}

		public void KillAllConnections(string instanceName, string databaseName)
		{
			var sql = $@"DECLARE @DatabaseName nvarchar(50)=N'{databaseName}'
DECLARE @SQL varchar(max)
SELECT @SQL = COALESCE(@SQL,'') + 'Kill ' + Convert(varchar, SPId) + ';'
FROM MASTER..SysProcesses
WHERE DBId = DB_ID(@DatabaseName) AND SPId <> @@SPId
EXEC(@SQL)";

			ExecuteNonQueryRawSql(instanceName, sql);
		}

		public IEnumerable<T> QuerySqlRaw<T>(string instanceName, string sql, object parameter = null)
		{
			var connectionString = GetConnectionString(instanceName);
			using var myConn = new SqlConnection(connectionString);
			return myConn.Query<T>(sql, parameter);
		}
		public void StartInstance(string instanceName)
		{
			ExecuteSqlLocalDbExe(@$"start ""{instanceName}""");
		}

		public void StopInstance(string instanceName)
		{
			ExecuteSqlLocalDbExe(@$"stop ""{instanceName}""");
		}

		private string GetConnectionString(string instanceName)
		{
			return $"Server=(localdb)\\{instanceName};Integrated security=SSPI;database=master";
		}
	}
}