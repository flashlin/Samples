using CloneSqlServer.Kit;
using FluentAssertions;

namespace DbSchemaExtractTests;

public class Tests
{
    private SqlDbContext _localDb;

    [SetUp]
    public async Task Setup()
    {
        var connectionString = SqlDbContext.BuildConnectionString("127.0.0.1:1433", 
            "YourSa", 
            "YourStrongPassword");

        _localDb = new SqlDbContext();
        await _localDb.OpenAsync(connectionString);

        await _localDb.ExecuteAsync("""
                                   IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'test')
                                   BEGIN
                                       CREATE DATABASE [test];
                                   END
                                   """);

        await _localDb.ExecuteAsync("""
                                   USE [test];
                                   
                                   -- Create Customer table
                                   IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Customer' AND xtype='U')
                                   BEGIN
                                       CREATE TABLE Customer (
                                           id INT IDENTITY(1,1) PRIMARY KEY,
                                           name NVARCHAR(100) NOT NULL,
                                           birth DATE NULL
                                       );
                                   END
                                   
                                   -- Create Product table
                                   IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Product' AND xtype='U')
                                   BEGIN
                                       CREATE TABLE Product (
                                           id INT IDENTITY(1,1) PRIMARY KEY,
                                           CustomerId INT NOT NULL,
                                           ProductName NVARCHAR(200) NOT NULL,
                                           Price DECIMAL(10,2) NOT NULL,
                                           BuyDate DATETIME NOT NULL DEFAULT GETDATE()
                                       );
                                   END
                                   
                                   -- Create foreign key constraint
                                   IF NOT EXISTS (SELECT * FROM sys.foreign_keys WHERE name = 'FK_Product_Customer')
                                   BEGIN
                                       ALTER TABLE Product
                                       ADD CONSTRAINT FK_Product_Customer
                                       FOREIGN KEY (CustomerId) REFERENCES Customer(id);
                                   END
                                   """);
         
                 
    }

    [TearDown]
    public async Task TearDown()
    {
        await _localDb.DisposeAsync();
    }

    [Test]
    public async Task TableSchema()
    {
        var tableSchemaList = await _localDb.QueryTableSchemaAsync();
        tableSchemaList.Should().BeEquivalentTo(
            new List<TableSchema>());
    }
}