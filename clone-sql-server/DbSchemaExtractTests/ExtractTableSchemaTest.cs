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
        tableSchemaList.Should().BeEquivalentTo(new List<TableSchema>
        {
            new TableSchema
            {
                Name = "Customer",
                Fields = new List<FieldSchema>
                {
                    new FieldSchema
                    {
                        Name = "id",
                        DataType = "int",
                        DataSize = 4,
                        DataScale = 0,
                        IsNullable = false,
                        IsPrimaryKey = true,
                        IsIdentity = true,
                        DefaultValue = "",
                        Description = ""
                    },
                    new FieldSchema
                    {
                        Name = "name",
                        DataType = "nvarchar",
                        DataSize = 200,
                        DataScale = 0,
                        IsNullable = false,
                        IsPrimaryKey = false,
                        IsIdentity = false,
                        DefaultValue = "",
                        Description = ""
                    },
                    new FieldSchema
                    {
                        Name = "birth",
                        DataType = "date",
                        DataSize = 3,
                        DataScale = 0,
                        IsNullable = true,
                        IsPrimaryKey = false,
                        IsIdentity = false,
                        DefaultValue = "",
                        Description = ""
                    }
                }
            },
            new TableSchema
            {
                Name = "Product",
                Fields = new List<FieldSchema>
                {
                    new FieldSchema
                    {
                        Name = "id",
                        DataType = "int",
                        DataSize = 4,
                        DataScale = 0,
                        IsNullable = false,
                        IsPrimaryKey = true,
                        IsIdentity = true,
                        DefaultValue = "",
                        Description = ""
                    },
                    new FieldSchema
                    {
                        Name = "CustomerId",
                        DataType = "int",
                        DataSize = 4,
                        DataScale = 0,
                        IsNullable = false,
                        IsPrimaryKey = false,
                        IsIdentity = false,
                        DefaultValue = "",
                        Description = ""
                    },
                    new FieldSchema
                    {
                        Name = "ProductName",
                        DataType = "nvarchar",
                        DataSize = 400,
                        DataScale = 0,
                        IsNullable = false,
                        IsPrimaryKey = false,
                        IsIdentity = false,
                        DefaultValue = "",
                        Description = ""
                    },
                    new FieldSchema
                    {
                        Name = "Price",
                        DataType = "decimal",
                        DataSize = 9,
                        DataScale = 2,
                        IsNullable = false,
                        IsPrimaryKey = false,
                        IsIdentity = false,
                        DefaultValue = "",
                        Description = ""
                    },
                    new FieldSchema
                    {
                        Name = "BuyDate",
                        DataType = "datetime",
                        DataSize = 8,
                        DataScale = 3,
                        IsNullable = false,
                        IsPrimaryKey = false,
                        IsIdentity = false,
                        DefaultValue = "(getdate())",
                        Description = ""
                    }
                }
            }
        });
    }
}