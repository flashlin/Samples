using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using FluentAssertions;
using Microsoft.Extensions.Options;
using QueryKits.Services;
using T1.Standard.Data.SqlBuilders;
using T1.Standard.DynamicCode;

namespace QueryKitsTests;

public class Tests
{
    private QueryService _sut = null!;
    private ReportDbContext _dbContext = null!;

    [SetUp]
    public void Setup()
    {
        //_dbContext = new ReportDbContext(new SqlMemoryDbContextOptionsFactory());
        var dbConfig = Options.Create(new DbConfig
        {
            ConnectionString =
                "Data Source=127.0.0.1,4331;User ID=sa;Password=Passw0rd!;Initial Catalog=QueryDb;TrustServerCertificate=true;"
        });
        _dbContext = new ReportDbContext(new DbContextOptionsFactory(dbConfig));
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
        var allTableNames = _dbContext.GetAllTableNames().Select(x => x.ToLower()).ToList();
        if (!allTableNames.Contains("customer"))
        {
            _dbContext.CreateTableByEntity(typeof(CustomerEntity));
        }

        if (!allTableNames.Contains("extracustomer"))
        {
            _dbContext.CreateTableByEntity(typeof(ExtraCustomerEntity));
        }

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

        var leftTable = _dbContext.SqlBuilder.GetTableInfo(typeof(CustomerEntity));
        var rightTable = _dbContext.SqlBuilder.GetTableInfo(typeof(ExtraCustomerEntity));
        _sut.MergeTable(new MergeTableRequest
        {
            LeftTable = leftTable,
            RightTable = rightTable,
            LeftJoinKeys = new List<TableColumnInfo>
            {
                leftTable.Columns.First(x => x.Name == "Id")
            },
            RightJoinKeys = new List<TableColumnInfo>
            {
                rightTable.Columns.First(x => x.Name == "CustomerId")
            },
            TargetTableName = "M1",
            MergeType = MergeType.InnerJoin
        });

        var actual = _dbContext.Query<MergeEntity>("SELECT * FROM M1")
            .ToList();
    }

    private void AddEntity(object data)
    {
        var entityType = data.GetType();
        var sql = CreateInsertTableStatement(entityType);
        _dbContext.Execute(sql, data);
    }

    private string CreateInsertTableStatement(Type entityType)
    {
        var sqlBuilder = _dbContext.SqlBuilder;
        var table = sqlBuilder.GetTableInfo(entityType);
        var propertyNames = sqlBuilder.GetTableProperties(entityType)
            .Join(table.Columns.Where(c => !c.IsAutoIncrement),
                property => property.ColumnInfo.Name,
                column => column.Name,
                (prop, column) => prop.Name)
            .ToList();
        var sql = sqlBuilder.CreateInsertStatement(table, propertyNames);
        return sql;
    }
}

public class MergeEntity
{
    public int LeftId { get; set; }
    public string Name { get; set; }
    public DateTime Birth { get; set; }
    public int RightId { get; set; }
    public int CustomerId { get; set; }
    public string Addr { get; set; }
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