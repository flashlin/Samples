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
                                   IF EXISTS (SELECT name FROM sys.databases WHERE name = N'test')
                                   BEGIN
                                       ALTER DATABASE [test] SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
                                       DROP DATABASE [test];
                                   END
                                   CREATE DATABASE [test];
                                   """);

        await _localDb.ExecuteAsync("""
                                   USE [test];

                                   -- Drop and recreate BProduct table
                                   IF OBJECT_ID('BProduct', 'U') IS NOT NULL
                                       DROP TABLE BProduct;
                                   
                                   CREATE TABLE BProduct (
                                       id INT IDENTITY(1,1) PRIMARY KEY,
                                       CustomerId INT NOT NULL,
                                       ProductName NVARCHAR(200) NOT NULL,
                                       Price DECIMAL(10,2) NOT NULL,
                                       BuyDate DATETIME NOT NULL DEFAULT GETDATE()
                                   );
                                   
                                   -- Drop and recreate Customer table
                                   IF OBJECT_ID('Customer', 'U') IS NOT NULL
                                       DROP TABLE Customer;
                                   
                                   CREATE TABLE Customer (
                                       id INT IDENTITY(1,1) PRIMARY KEY,
                                       name NVARCHAR(100) NOT NULL,
                                       birth DATE NULL
                                   );
                                   
                                   -- Drop and recreate loginLog table
                                   IF OBJECT_ID('loginLog', 'U') IS NOT NULL
                                       DROP TABLE loginLog;
                                   
                                   CREATE TABLE loginLog (
                                       id INT IDENTITY(1,1) PRIMARY KEY,
                                       loginName NVARCHAR(100) NOT NULL,
                                       loginTime DATETIME NOT NULL DEFAULT GETDATE()
                                   );
                                   
                                   -- Create foreign key constraint
                                   IF NOT EXISTS (SELECT * FROM sys.foreign_keys WHERE name = 'FK_BProduct_Customer')
                                   BEGIN
                                       ALTER TABLE BProduct
                                       ADD CONSTRAINT FK_BProduct_Customer
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
                Name = "BProduct",
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
            },
            new TableSchema
            {
                Name = "loginLog",
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
                        Name = "loginName",
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
                        Name = "loginTime",
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

    [Test]
    public async Task TableDependency()
    {
        var tables = await _localDb.QueryTableSchemaAsync();
        var fkList = await _localDb.QueryForeignKeyAsync();
        var tableList = _localDb.GetTablesInDependencyOrder(tables, fkList);
        
        // Verify the correct dependency order
        tableList.Should().HaveCount(3);
        
        // loginLog has no dependencies, so it can be at any position after Customer
        tableList.Should().Contain(t => t.Name == "loginLog");
        
        // Verify that Customer comes before BProduct (dependency order)
        var customerIndex = tableList.FindIndex(t => t.Name == "Customer");
        var bproductIndex = tableList.FindIndex(t => t.Name == "BProduct");
        customerIndex.Should().BeLessThan(bproductIndex);
    }
}