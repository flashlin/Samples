using SqlLocalDataTests.Repositories;
using System;
using System.IO;
using System.Linq;
using Microsoft.EntityFrameworkCore;
using T1.SqlLocalData;
using T1.SqlLocalData.Extensions;
using Xunit;

namespace SqlLocalDataTests
{
	public class FastCreateTest : IDisposable
	{
		private string _databaseFile = @"D:\Demo\fasttest.mdf";
		private string _databaseName = "fasttest";
		private string _instanceName = "local_fast";
		private MyDbContext _myDb;
		private readonly SqlLocalDb _localDb = new SqlLocalDb();

		public FastCreateTest()
		{
			_localDb.EnsureInstanceCreated(_instanceName);
			CreateDatabase();
			_myDb = new MyDbContext(_instanceName, _databaseName);
		}

		[Fact]
		public void execute_raw_sql()
		{
			_myDb.Database.ExecuteSqlRaw(@"CREATE TABLE customer (id INT PRIMARY KEY, name VARCHAR(50))");
			_myDb.Database.ExecuteSqlRaw(@"INSERT customer(id,name) VALUES (1,'Flash'),(3,'Jack'),(4,'Mary')");

			var customer = _myDb.Customers
				.First(x => x.Id == 3);

			Assert.Equal("Jack",customer.Name);
		}
		
		[Fact]
		public void execute_raw_sql_from_file()
		{
			_myDb.ExecuteSqlRawFromFile("./Contents/CreateTable.sql");
			_myDb.ExecuteSqlRawFromFile("./Contents/MyGetCustomer.sql");

			var customer = _myDb.Customers
				.First(x => x.Id == 3);

			Assert.Equal("Jack",customer.Name);
		}

		
		[Fact]
		public void execute_store_procedure()
		{
			_myDb.Database.ExecuteSqlRaw(@"CREATE TABLE customer (id INT PRIMARY KEY, name VARCHAR(50))");
			_myDb.Database.ExecuteSqlRaw(@"INSERT customer(id,name) VALUES (1,'Flash'),(3,'Jack'),(4,'Mary')");
			_myDb.Database.ExecuteSqlRaw(@"CREATE PROC MyGetCustomer 
				@id INT AS 
				BEGIN 
					SET NOCOUNT ON; 
					select name from customer 
					WHERE id=@id 
				END");

			var customer = _myDb.QuerySqlRaw<CustomerEntity>("EXEC MyGetCustomer @id", new
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
			_localDb.ForceDropDatabase(_instanceName, _databaseName);
			_localDb.DeleteDatabaseFile(_databaseFile);
			_localDb.CreateDatabase(_instanceName, _databaseFile);
		}
	}
}