using FluentAssertions;
using Microsoft.EntityFrameworkCore;
using SqlSharpLit;

namespace SqlSharpTests;

public class Tests
{
    private DynamicDbContext _db;

    [SetUp]
    public void Setup()
    {
        _db = new DynamicDbContext(DynamicDbContext.CreateInMemoryDbContextOptions());
        _db.Database.ExecuteSql($"""
                                 CREATE TABLE [dbo].[Customer] (
                                     [Id] INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
                                     [Name] NVARCHAR(50) NOT NULL,
                                     [Email] NVARCHAR(50) NOT NULL
                                 )
                                 """);
        _db.Database.ExecuteSql($"""
                                 INSERT INTO [dbo].[Customer] ([Name], [Email]) VALUES ('John Doe', 'test1@mail.com')
                                 INSERT INTO [dbo].[Customer] ([Name], [Email]) VALUES ('Mary', 'test2@mail.com')
                                 """);
    }

    [Test]
    public void Test1()
    {
        var fields = _db.GetTableSchema("Customer");
        var data = _db.GetTopNTableData(1, "Customer", fields, null);
        data.Should().BeEquivalentTo([
            new Dictionary<string, string>()
            {
                ["Id"] = "1",
                ["Name"] = "John Doe",
                ["Email"] = "test1@mail.com"
            }
        ]);
    }
}