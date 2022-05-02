using System;
using System.Runtime.InteropServices;
using Microsoft.EntityFrameworkCore;
using SqlLocalDbTests.Repositories;
using T1.SqlLocalData;

namespace SqlLocalDbTests;

public class InitializeFixture : IDisposable
{
	private string _databaseName = "Northwind";
	private string _instanceName = "localtest";

	public InitializeFixture()
	{
		if (GetOSPlatform() == OSPlatform.Linux)
		{
			LocalDb = new LinuxLocalDb();
		}
		else
		{
			LocalDb = new SqlLocalDb(@"D:\Demo");
		}

		InitializeSqlLocalDbInstance();
	}

	public ISqlLocalDb LocalDb { get; }

	public void CreateSp()
	{
		var myDb = GetMyDb();
		myDb.Database.ExecuteSqlRaw(@"CREATE PROC MyGetCustomer 
	 @id INT AS 
	 BEGIN 
	 	SET NOCOUNT ON; 
	 	select name from customer 
	 	WHERE id=@id 
	 END");
	}

	public void CreateTable()
	{
		var myDb = GetMyDb();
		myDb.Database.ExecuteSqlRaw(@"CREATE TABLE customer (id INT PRIMARY KEY, name VARCHAR(50))");
		myDb.Database.ExecuteSqlRaw(@"INSERT customer(id,name) VALUES (1,'Flash'),(3,'Jack'),(4,'Mary')");
	}

	public void Dispose()
	{
	}
	public MyDbContext GetMyDb()
	{
		return new MyDbContext(LocalDb.GetDatabaseConnectionString(_instanceName, _databaseName));
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
		LocalDb.EnsureInstanceCreated(_instanceName);
		LocalDb.ForceDropDatabase(_instanceName, _databaseName);
		LocalDb.DeleteDatabaseFile(_databaseName);
		LocalDb.CreateDatabase(_instanceName, _databaseName);
	}
}