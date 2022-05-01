using Dapper;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.IO;
using System.Linq;

namespace T1.SqlLocalData
{
    public class SqlLocalDb
    {
        private readonly string _dataFolder;
        readonly string _sqllocaldbexe = @"C:\Program Files\Microsoft SQL Server\150\Tools\Binn\SqlLocalDB.exe";

        public SqlLocalDb(string dataFolder)
        {
            _dataFolder = dataFolder;
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

        public void DropDatabase(string instanceName, string databaseName)
        {
            var sql = $"DROP DATABASE IF EXISTS {databaseName}";
            ExecuteNonQueryRawSql(instanceName, sql);
        }

        public void ForceDropDatabase(string instanceName, string databaseName)
        {
            KillAllConnections(instanceName, databaseName);
            DropDatabase(instanceName, databaseName);
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

        public void ExecuteNonQueryRawSql(string instanceName, string sql, object parameter = null)
        {
            //$"Server=localhost;Integrated security=SSPI;database=master;AttachDBFilename=|DataDirectory|\\{databaseName}.mdf");
            using var myConn = new SqlConnection(GetConnectionString(instanceName));
            myConn.ExecuteScalar(sql, parameter);
        }

        public string ExecuteSqlLocalDbExe(string arguments)
        {
            var p = new ProcessHelper();
            return p.Execute(_sqllocaldbexe, arguments);
        }

        public void ForceStopInstance(string instanceName)
        {
            ExecuteSqlLocalDbExe(@$"stop ""{instanceName}"" -i -k");
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

            return QueryRawSql<bool>(instanceName, sql, new
            {
                dbname = databaseName
            }).First();
        }

        public bool IsInstanceExists(string instanceName)
        {
            return GetInstanceNames()
                .Any(x => x == instanceName);
        }

        public IEnumerable<T> QueryRawSql<T>(string instanceName, string sql, object parameter = null)
        {
            using var myConn = new SqlConnection(GetConnectionString(instanceName));
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

        public void EnsureInstanceCreated(string instanceName)
        {
            if (IsInstanceExists(instanceName))
            {
                return;
            }

            CreateInstance(instanceName);
            StartInstance(instanceName);
        }
    }
}