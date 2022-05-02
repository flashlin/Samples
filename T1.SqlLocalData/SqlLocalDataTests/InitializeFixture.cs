using System;
using Microsoft.EntityFrameworkCore;
using SqlLocalDataTests.Repositories;

namespace SqlLocalDataTests;

public class InitializeFixture : IDisposable
{
    public MyDbContext MyDb { get; }

    public InitializeFixture()
    {
        MyDb = new MyDbContext("Server=db;User Id=sa;Password=1Secure*Password1;");
        CreateTable1();
    }

    public void Dispose()
    {
    }
		
    private void CreateTable1()
    {
        MyDb.Database.ExecuteSqlRaw(@"CREATE TABLE customer (id INT PRIMARY KEY, name VARCHAR(50))");
        MyDb.Database.ExecuteSqlRaw(@"INSERT customer(id,name) VALUES (1,'Flash'),(3,'Jack'),(4,'Mary')");
        MyDb.Database.ExecuteSqlRaw(@"CREATE PROC MyGetCustomer 
				@id INT AS 
				BEGIN 
					SET NOCOUNT ON; 
					select name from customer 
					WHERE id=@id 
				END");
    }
		
    private void CreateTable2()
    {
        MyDb.ExecuteSqlRawFromFile("./Contents/CreateTable.sql");
        MyDb.ExecuteSqlRawFromFile("./Contents/MyGetCustomer.sql");
    }
}