using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using FluentAssertions;
using QueryKits.Services;

namespace QueryKitsTests;

public class Tests
{
    private QueryService _sut = null!;
    private ReportDbContext _dbContext = null!;

    [SetUp]
    public void Setup()
    {
        _dbContext = new ReportDbContext(new SqliteMemoryDbContextOptionsFactory());
        _sut = new QueryService(_dbContext);
    }

    [Test]
    public void CreateTable()
    {
        _dbContext.CreateTableByEntity(typeof(CustomerEntity));
        _dbContext.CreateTableByEntity(typeof(ExtraCustomerEntity));
        
        var tables = _dbContext.GetAllTableNames();
        tables.Count().Should().Be(2);
    }
}

[Table("Customer")]
public class CustomerEntity
{
    [Key]
    [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
    public int Id { get; set; } 
    public string Name { get; set; }
    public DateTime Birth { get; set; }
}

[Table("ExtraCustomer")]
public class ExtraCustomerEntity
{
    [Key]
    [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
    public int Id { get; set; } 
    public int CustomerId { get; set; }
    [Column("Addr", TypeName = "NVARCHAR(50)")]
    public string Address { get; set; }
}