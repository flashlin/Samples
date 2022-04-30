using SqlLocalDataTests.Repositories;
using System;
using System.IO;
using System.Linq;
using Microsoft.EntityFrameworkCore;
using T1.SqlLocalData;
using Xunit;

namespace SqlLocalDataTests
{
	public class CreateTest : IDisposable
	{
		private string _databaseFile = @"D:\Demo\test.mdf";
		private string _instanceName = "localtest";
		private readonly SqlLocalDb _localDb = new SqlLocalDb();

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
			var myDb = new MyDbContext();
			myDb.CreateTable(typeof(CustomerEntity));
		}
		
		[Fact]
		public void execute_raw_sql()
		{
			var myDb = new MyDbContext();
			myDb.Database.ExecuteSqlRaw(@"CREATE TABLE customer (id INT PRIMARY KEY, name VARCHAR(50))");
			myDb.Database.ExecuteSqlRaw(@"INSERT customer(id,name) VALUES (1,'Flash'),(3,'Jack'),(4,'Mary')");

			var customer = myDb.Customers
				.First(x => x.Id == 3);

			Assert.Equal("Jack",customer.Name);
		}
		
		
		[Fact]
		public void execute_store_procedure()
		{
			var myDb = new MyDbContext();
			myDb.Database.ExecuteSqlRaw(@"CREATE TABLE customer (id INT PRIMARY KEY, name VARCHAR(50))");
			myDb.Database.ExecuteSqlRaw(@"INSERT customer(id,name) VALUES (1,'Flash'),(3,'Jack'),(4,'Mary')");
			myDb.Database.ExecuteSqlRaw(@"CREATE PROC MyGetCustomer @id INT AS BEGIN SET NOCOUNT ON; select name from customer WHERE id=@id END");

			var customer = myDb.QueryRawSql<CustomerEntity>("EXEC MyGetCustomer @id", new
			{
				id = 3
			}).First();

			Assert.Equal("Jack",customer.Name);
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