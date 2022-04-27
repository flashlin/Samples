using SqlLocalDataTests.Repositories;
using System;
using System.IO;
using T1.SqlLocalData;
using Xunit;

namespace SqlLocalDataTests
{
	public class CreateTest : IDisposable
	{
		private string _databaseFile = @"D:\Demo\test.mdf";
		private string _instanceName = "localtest";
		private SqlLocalDb _localDb = new SqlLocalDb();

		public CreateTest()
		{
			CreateInstance();
			CreateDatabase();
		}

		[Fact]
		public void database_exists()
		{
			var dbExists = _localDb.IsDatabaseExists(_instanceName, "test");
			Assert.True(dbExists);
		}

		[Fact]
		public void create_table()
		{
			var mydb = new MyDbContext();
			mydb.EnsureTableCreated(typeof(CustomerEntity));
		}

		public void Dispose()
		{
			//_localDb.DeleteInstance();
		}

		private void CreateDatabase()
		{
			_localDb.DeleteDatabaseFile(_databaseFile);
			_localDb.CreateDatabase(_instanceName, _databaseFile);
		}

		private void CreateInstance()
		{
			if (_localDb.IsInstanceExists(_instanceName))
			{
				_localDb.StopInstance(_instanceName);
				_localDb.DeleteInstance(_instanceName);
			}
			_localDb.CreateInstance(_instanceName);
			_localDb.StartInstance(_instanceName);
		}
	}
}