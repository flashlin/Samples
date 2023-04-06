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
    
    
    [Test]
    public void MergeTable()
    {
        _dbContext.CreateTableByEntity(typeof(CustomerEntity));
        _dbContext.CreateTableByEntity(typeof(ExtraCustomerEntity));
        AddEntity(new CustomerEntity
        {
            Name = "flash",
            Birth = new DateTime(2019, 01, 01)
        });
        AddEntity(new CustomerEntity
        {
            Name = "mary",
            Birth = new DateTime(2019, 01, 02)
        });
        AddEntity(new ExtraCustomerEntity
        {
            CustomerId = 1,
            Address = "Taipei"
        });
        AddEntity(new ExtraCustomerEntity
        {
            CustomerId = 2,
            Address = "Taihju"
        });

        _sut.MergeTable(new MergeTableRequest
        {
            LeftTable = new TableInfo
            {
                Name = "Customer",
                Columns = new []
                {
                    new TableColumnInfo
                    {
                        IsKey = false,
                        IsAutoIncrement = false,
                        Name = "Id",
                        DataType = "INT",
                        Size = 0,
                        Precision = 0,
                        Scale = 0
                    }
                }.ToList(),
            },
            RightTable = new TableInfo()
            {
                Name = "ExtraCustomer",
                Columns = new []
                {
                    new TableColumnInfo
                    {
                        Name = "CustomerId",
                        DataType = "INT",
                    }
                }.ToList(),
            },
            TargetTableName = "M1",
            MergeType = MergeType.InnerJoin
        });
    }

    private void AddEntity(object data)
    {
        var entityType = data.GetType();
        var sql = CreateTableStatement(entityType);
        _dbContext.ExecuteRawSql(sql, data);
    }

    private string CreateTableStatement(Type entityType)
    {
        var sqlBuilder = _dbContext.SqlBuilder;
        var table = sqlBuilder.GetTableInfo(entityType);
        var sql = sqlBuilder.CreateInsertStatement(table);
        return sql;
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