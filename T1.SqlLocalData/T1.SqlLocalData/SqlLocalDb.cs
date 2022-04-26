using System;
using System.Data;
using System.Data.SqlClient;
using System.IO;

namespace T1.SqlLocalData
{
	public class SqlLocalDb
	{


		public void CreateDatabase(string databaseMdfFile)
		{
			var databaseName = Path.GetFileNameWithoutExtension(databaseMdfFile);
			var databasePath = Path.GetDirectoryName(databaseMdfFile);

			using var myConn = new SqlConnection("Server=(localdb)\\.;Integrated security=SSPI;database=master");

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
			using var myConn = new SqlConnection($"Server=localhost;Integrated security=SSPI;database=master;AttachDBFilename=|{dataDirectory}|\\{databaseName}.mdf");
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
}
