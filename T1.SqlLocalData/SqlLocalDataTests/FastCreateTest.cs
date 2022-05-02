using SqlLocalDataTests.Repositories;
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.EntityFrameworkCore;
using T1.SqlLocalData;
using T1.SqlLocalData.Extensions;
using Xunit;

namespace SqlLocalDataTests
{
	public class FastCreateTest
	{
		private readonly SqlLocalDb _localDb = new SqlLocalDb(@"D:\Demo");
		private string _databaseName = "fasttest";
		private string _instanceName = "local_fast";
		private MyDbContext _myDb;

		public FastCreateTest()
		{
			InitializeSqlLocalDbInstance();

			var windowsSqlLocalDbConnectionString =
				 $"Server=(localdb)\\{_instanceName};Integrated security=SSPI;database={_databaseName};";
			var linuxConnectionString = "Server=db;Database=Northwind;User=sa;Password=1Secure*Password1;";

			var connectionString = GetOSPlatform() == OSPlatform.Linux
				 ? linuxConnectionString
				 : windowsSqlLocalDbConnectionString;

			_myDb = new MyDbContext(connectionString);
		}

		[Fact]
		public void execute_raw_sql()
		{
			_myDb.Database.ExecuteSqlRaw(@"CREATE TABLE customer (id INT PRIMARY KEY, name VARCHAR(50))");
			_myDb.Database.ExecuteSqlRaw(@"INSERT customer(id,name) VALUES (1,'Flash'),(3,'Jack'),(4,'Mary')");

			var customer = _myDb.Customers
				 .First(x => x.Id == 3);

			Assert.Equal("Jack", customer.Name);
		}

		[Fact]
		public void execute_raw_sql_from_file()
		{
			_myDb.ExecuteSqlRawFromFile("./Contents/CreateTable.sql");
			_myDb.ExecuteSqlRawFromFile("./Contents/MyGetCustomer.sql");

			var customer = _myDb.Customers
				 .First(x => x.Id == 3);

			Assert.Equal("Jack", customer.Name);
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

			Assert.Equal("Jack", customer.Name);
		}

		private OSPlatform GetOSPlatform()
		{
			if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
			{
				return OSPlatform.Linux;
			}

			if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
			{
				return OSPlatform.OSX;
			}

			if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
			{
				return OSPlatform.Windows;
			}

			throw new Exception("Cannot determine operating system!");
		}

		private void InitializeSqlLocalDbInstance()
		{
			if(GetOSPlatform() != OSPlatform.Windows)
			{
				return;
			}

			_localDb.EnsureInstanceCreated(_instanceName);
			_localDb.ForceDropDatabase(_instanceName, _databaseName);
			_localDb.DeleteDatabaseFile(_databaseName);
			_localDb.CreateDatabase(_instanceName, _databaseName);
		}
	}
}