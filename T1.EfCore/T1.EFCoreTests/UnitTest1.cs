using FluentAssertions;
using Microsoft.EntityFrameworkCore;
using T1.EfCore;

namespace T1.EFCoreTests;

public class Tests
{
    private TestDbContext _db;

    [SetUp]
    public void Setup()
    {
        _db = new TestDbContext();
    }

    [Test]
    public void Test1()
    {
        GivenCreateCustomerTable();

        _db.Upsert(new CustomerEntity
        {
            Id = 1,
            Name = "flash"
        }).On(x => x.Id)
            .Execute();

        var customers = _db.Customer.ToArray();
        customers.Should().BeEquivalentTo([
            new CustomerEntity
            {
                Id = 1,
                Name = "flash"
            }
        ]);
    }

    private void GivenCreateCustomerTable()
    {
        _db.Database.ExecuteSqlRaw($"""
                                    DROP TABLE IF EXISTS [dbo].[Customer];
                                    CREATE TABLE [dbo].[Customer] (
                                      [Id] [int] NOT NULL,
                                      [Name] [nvarchar](50) NOT NULL,
                                      CONSTRAINT [PK_Customer] PRIMARY KEY ([Id])
                                    )
                                    """);
    }

    [TearDown]
    public void TearDown()
    {
        _db.Dispose();
    }
}