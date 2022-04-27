using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;

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

        public bool IsInstanceExists(string instanceName)
        {
            return GetInstanceNames()
                .Any(x => x == instanceName);
        }

        public void StartInstance(string instanceName)
        {
            ExecuteSqlLocalDbExe(@$"START ""{instanceName}""");
        }

        public string ExecuteSqlLocalDbExe(string arguments)
        {
            var p = new ProcessHelper();
            return p.Execute(_sqllocaldbexe, arguments);
        }


        public void CreateDatabase(string instanceName, string databaseMdfFile)
        {
            var databaseName = Path.GetFileNameWithoutExtension(databaseMdfFile);
            var databasePath = Path.GetDirectoryName(databaseMdfFile);

            using var myConn =
                new SqlConnection($"Server=(localdb)\\{instanceName};Integrated security=SSPI;database=master");

            var str = "CREATE DATABASE MyDatabase ON PRIMARY " +
                      $"(NAME = {databaseName}_Data, " +
                      $"FILENAME = '{databaseMdfFile}', " +
                      "SIZE = 2MB, MAXSIZE = 10MB, FILEGROWTH = 10%)" +
                      $"LOG ON (NAME = {databaseName}_Log, " +
                      $"FILENAME = '{databasePath}\\{databaseName}.ldf', " +
                      "SIZE = 1MB, MAXSIZE = 5MB, FILEGROWTH = 10%)";

            var myCommand = new SqlCommand(str, myConn);
            try
            {
                myConn.Open();
                myCommand.ExecuteNonQuery();
            }
            finally
            {
                if (myConn.State == ConnectionState.Open)
                {
                    myConn.Close();
                }
            }
        }

        public void ExecuteNonQueryRawSql(string sql)
        {
            var dataDirectory = @"D:\\Demo";
            var databaseName = "Test";
            using var myConn =
                new SqlConnection(
                    $"Server=localhost;Integrated security=SSPI;database=master;AttachDBFilename=|{dataDirectory}|\\{databaseName}.mdf");
            //Data Source=(LocalDb)\MSSQLLocalDB;Initial Catalog=aspnet-MvcMovie;Integrated Security=SSPI;
            //AttachDBFilename=|DataDirectory|\Movies.mdf

            var myCommand = new SqlCommand(sql, myConn);
            try
            {
                myConn.Open();
                myCommand.ExecuteNonQuery();
            }
            finally
            {
                if (myConn.State == ConnectionState.Open)
                {
                    myConn.Close();
                }
            }
        }
    }

    public class ProcessHelper
    {
        public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(30);

        public string Execute(string processFilename, string arguments)
        {
            var startInfo = new ProcessStartInfo(processFilename)
            {
                WindowStyle = ProcessWindowStyle.Hidden,
                Arguments = arguments,
                RedirectStandardError = true,
                RedirectStandardOutput = true,
                CreateNoWindow = true,
                UseShellExecute = false,
            };

            var outputStringBuilder = new StringBuilder();

            var process = new Process();
            process.StartInfo = startInfo;
            process.EnableRaisingEvents = false;
            process.OutputDataReceived += (sender, eventArgs) => outputStringBuilder.AppendLine(eventArgs.Data);
            process.ErrorDataReceived += (sender, eventArgs) => outputStringBuilder.AppendLine(eventArgs.Data);

            try
            {
                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
                //process.StandardOutput.ReadToEnd();
                var processExited = process.WaitForExit((int) Timeout.TotalMilliseconds);
                if (processExited == false)
                {
                    process.Kill();
                    throw new Exception("ERROR: Process took too long to finish");
                }

                if (process.ExitCode != 0)
                {
                    var output = outputStringBuilder.ToString();
                    throw new Exception("Process exited code: " + process.ExitCode + Environment.NewLine +
                                        "Output from process: " + outputStringBuilder.ToString());
                }
            }
            finally
            {
                process.Close();
            }

            return outputStringBuilder.ToString();
        }
    }
}