using FluentAssertions;
using Microsoft.EntityFrameworkCore;
using T1.EfCore;

namespace T1.EFCoreTests;

public class UpsertTest
{
    private TestDbContext _db;

    [SetUp]
    public void Setup()
    {
        _db = new TestDbContext();
    }
    
    
    [Test]
    public void UpsertRange()
    {
        GivenCreateCustomerTable();
        var existedData = new List<CustomerEntity>()
        {
            new()
            {
                Id = 2,
                Name = "Jack",
            },
        };

        _db.Customer.AddRange(existedData);
        _db.SaveChanges();
        
        var insertData = new List<CustomerEntity>()
        {
            new()
            {
                Id = 1,
                Name = "Flash",
            },
            new()
            {
                Id = 2,
                Name = "Jack",
            },
            new()
            {
                Id = 3,
                Name = "Mark",
            }
        };
        _db.UpsertRange(insertData)
            .On(x=>x.Id)
            .Execute();
        
        _db.UpsertRange(insertData)
            .On(x=>x.Id)
            .Execute();
        
        var customers = _db.Customer.ToArray();
        customers.Should().BeEquivalentTo([
            new CustomerEntity
            {
                Id = 1,
                Name = "Flash"
            },
            new CustomerEntity
            {
                Id = 2,
                Name = "Jack"
            },
            new CustomerEntity
            {
                Id = 3,
                Name = "Mark"
            }
        ]);
    }

    [Test]
    public void BulkInsertRange()
    {
        GivenCreateCustomerTable();
        var data = new List<CustomerEntity>()
        {
            new()
            {
                Id = 1,
                Name = "Flash",
            },
            new()
            {
                Id = 2,
                Name = "Jack",
            },
            new()
            {
                Id = 3,
                Name = "Mark",
            }
        };
        _db.BulkInsert(data)
            .Execute();
        
        var customers = _db.Customer.ToArray();
        customers.Should().BeEquivalentTo([
            new CustomerEntity
            {
                Id = 1,
                Name = "Flash"
            },
            new CustomerEntity
            {
                Id = 2,
                Name = "Jack"
            },
            new CustomerEntity
            {
                Id = 3,
                Name = "Mark"
            }
        ]);

    }

    
    [Test]
    public void Empty()
    {
        GivenCreateCustomerTable();
        WhenUpsert(
            new CustomerEntity
            {
                Id = 1,
                Name = "flash"
            });

        var customers = _db.Customer.ToArray();
        customers.Should().BeEquivalentTo([
            new CustomerEntity
            {
                Id = 1,
                Name = "flash"
            }
        ]);
    }


    [Test]
    public void DataExistsOn2()
    {
        GivenCreateCustomerTable();
        _db.Customer.Add(new CustomerEntity
        {
            Id = 1,
            Name = "flash"
        });

        
        _db.Upsert(new CustomerEntity
        {
            Id = 1,
            Name = "flash"
        }, new CustomerEntity
        {
            Id = 2,
            Name = "jack"
        }).On(x => new {x.Id, x.Name}) 
            .Execute();

        var customers = _db.Customer.ToArray();
        customers.Should().BeEquivalentTo([
            new CustomerEntity
            {
                Id = 1,
                Name = "flash"
            },
            new CustomerEntity
            {
                Id = 2,
                Name = "jack"
            }
        ]);
    }
    
    [Test]
    public void DataExists()
    {
        GivenCreateCustomerTable();
        _db.Customer.Add(new CustomerEntity
        {
            Id = 1,
            Name = "flash"
        });

        WhenUpsert(new CustomerEntity
        {
            Id = 1,
            Name = "flash"
        }, new CustomerEntity
        {
            Id = 2,
            Name = "jack"
        });


        var customers = _db.Customer.ToArray();
        customers.Should().BeEquivalentTo([
            new CustomerEntity
            {
                Id = 1,
                Name = "flash"
            },
            new CustomerEntity
            {
                Id = 2,
                Name = "jack"
            }
        ]);
    }


    [Test]
    public void UpsertArray()
    {
        GivenCreateCustomerTable();
        _db.Customer.Add(new CustomerEntity
        {
            Id = 1,
            Name = "flash"
        });

        WhenUpsert([new CustomerEntity
        {
            Id = 1,
            Name = "flash"
        }, new CustomerEntity
        {
            Id = 2,
            Name = "jack"
        }]);

        var customers = _db.Customer.ToArray();
        customers.Should().BeEquivalentTo([
            new CustomerEntity
            {
                Id = 1,
                Name = "flash"
            },
            new CustomerEntity
            {
                Id = 2,
                Name = "jack"
            }
        ]);
    }


    private void WhenUpsert(params CustomerEntity[] entity)
    {
        _db.Upsert(entity).On(x => x.Id)
            .Execute();
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