using Dapper;
using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.IO;
using System.Linq;

namespace T1.SqlLocalData
{
	public class SqlLocalDb
	{
		const string _sqllocaldbexe = @"C:\Program Files\Microsoft SQL Server\150\Tools\Binn\SqlLocalDB.exe";

		public void CreateInstance(string instanceName)
		{
			ExecuteSqlLocalDbExe(@$"CREATE ""{instanceName}"" 15.0 -s");
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

		public void DeleteDatabaseFile(string databaseFile)
		{
			var databaseName = Path.GetFileNameWithoutExtension(databaseFile);
			var dataDirectory = Path.GetDirectoryName(databaseFile);
			File.Delete(databaseFile);
			File.Delete(Path.Combine(dataDirectory, $"{databaseName}.ldf"));
		}

		public bool IsInstanceExists(string instanceName)
		{
			return GetInstanceNames()
				 .Any(x => x == instanceName);
		}

		public void StartInstance(string instanceName)
		{
			ExecuteSqlLocalDbExe(@$"start ""{instanceName}""");
		}

		public void StopInstance(string instanceName)
		{
			ExecuteSqlLocalDbExe(@$"stop ""{instanceName}""");
		}

		public void ForceStopInstance(string instanceName)
		{
			ExecuteSqlLocalDbExe(@$"stop ""{instanceName}"" -i -k");
		}

		public void DeleteInstance(string instanceName)
		{
			ExecuteSqlLocalDbExe(@$"delete ""{instanceName}""");
		}

		public string ExecuteSqlLocalDbExe(string arguments)
		{
			var p = new ProcessHelper();
			return p.Execute(_sqllocaldbexe, arguments);
		}

		public bool IsDatabaseExists(string instanceName, string databaseName)
		{
			var sql = $@"if EXISTS(select dbid from (select DB_ID(@dbname) as dbid) t WHERE dbid IS NOT NULL)
select 1 
ELSE 
select 0";

			return QueryRawSql<bool>(instanceName, sql, new 
			{
				dbname = databaseName
			}).First();
		}


		public void CreateDatabase(string instanceName, string databaseMdfFile)
		{
			var databaseName = Path.GetFileNameWithoutExtension(databaseMdfFile);
			var databasePath = Path.GetDirectoryName(databaseMdfFile);

			var sql = $"CREATE DATABASE {databaseName} ON PRIMARY " +
						 $"(NAME={databaseName}_Data, FILENAME='{databaseMdfFile}', SIZE=2MB, MAXSIZE=10MB, FILEGROWTH= 10%) " +
						 $"LOG ON (NAME = {databaseName}_Log, FILENAME='{databasePath}\\{databaseName}.ldf', SIZE=1MB, MAXSIZE=5MB, FILEGROWTH=10%) " + 
						 "";

			ExecuteNonQueryRawSql(instanceName, sql);
		}

		public void DetachDatabase(string instanceName, string databaseName)
		{
			ExecuteNonQueryRawSql(instanceName, $"EXEC sp_detach_db '{databaseName}', 'true';");
		}

		public void ExecuteNonQueryRawSql(string instanceName, string sql, object parameter = null)
		{
			//$"Server=localhost;Integrated security=SSPI;database=master;AttachDBFilename=|DataDirectory|\\{databaseName}.mdf");
			using var myConn = new SqlConnection(GetConnectionString(instanceName));
			myConn.ExecuteScalar(sql, parameter);

			//var cmd = new SqlCommand(sql, myConn);
			//try
			//{
			//	myConn.Open();
			//	cmd.ExecuteNonQuery();
			//}
			//finally
			//{
			//	if (myConn.State == ConnectionState.Open)
			//	{
			//		myConn.Close();
			//	}
			//}
		}

		public IEnumerable<T> QueryRawSql<T>(string instanceName, string sql, object parameter = null)
		{
			using var myConn = new SqlConnection(GetConnectionString(instanceName));
			return myConn.Query<T>(sql, parameter);
		}

		private string GetConnectionString(string instanceName)
		{
			return $"Server=(localdb)\\{instanceName};Integrated security=SSPI;database=master";
		}
	}
}