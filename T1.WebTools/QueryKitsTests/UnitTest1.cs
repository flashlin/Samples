using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using FluentAssertions;
using Microsoft.Extensions.Options;
using QueryKits.Services;
using T1.SqlLocalData;
using T1.Standard.Data.SqlBuilders;
using T1.Standard.DynamicCode;

namespace QueryKitsTests;

public class Tests
{
    private string _instanceName = "local_db_instance";
    private string _databaseName = "QueryDb";
    private readonly SqlLocalDb _localDb = new SqlLocalDb(@"D:\Demo");

    private QueryService _sut = null!;
    private ReportDbContext _dbContext = null!;

    [SetUp]
    public void Setup()
    {
        _localDb.EnsureInstanceCreated(_instanceName);
        _localDb.ForceDropDatabase(_instanceName, _databaseName);
        _localDb.DeleteDatabaseFile(_databaseName);
        _localDb.CreateDatabase(_instanceName, _databaseName);


        //_dbContext = new ReportDbContext(new SqlMemoryDbContextOptionsFactory());
        var dbConfig = Options.Create(new DbConfig
        {
            ConnectionString =
                // "Data Source=127.0.0.1,4331;User ID=sa;Password=Passw0rd!;Initial Catalog=QueryDb;TrustServerCertificate=true;"
                "Server=(localdb)\\local_db_instance;Integrated security=SSPI;database=QueryDb;"
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
    public void MergeTableInto()
    {
        GiveTables();
        GiveData();

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

        var actual = ThenGetResult();

        actual.Should().BeEquivalentTo(new List<MergeEntity>()
        {
            new()
            {
                LeftId = 1,
                Name = "flash",
                Birth = new DateTime(2019, 1, 1),
                RightId = 1,
                CustomerId = 1,
                Addr = "Taipei"
            },
        });
    }
    
    [Test]
    public void MergeTableLeftInto()
    {
        GiveTables();
        GiveData();

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
            MergeType = MergeType.LeftJoin
        });

        var actual = ThenGetResult();

        actual.Should().BeEquivalentTo(new List<MergeEntity>()
        {
            new()
            {
                LeftId = 1,
                Name = "flash",
                Birth = new DateTime(2019, 1, 1),
                RightId = 1,
                CustomerId = 1,
                Addr = "Taipei"
            },
            new()
            {
                LeftId = 2,
                Name = "mary",
                Birth = new DateTime(2019, 1, 2),
                RightId = 0,
                CustomerId = 0,
                Addr = null
            },
        });
    }
    
    [Test]
    public void MergeTableLeftExcludeInto()
    {
        GiveTables();
        GiveData();

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
            MergeType = MergeType.LeftExcludeCrossJoin
        });

        var actual = ThenGetResult();

        actual.Should().BeEquivalentTo(new List<MergeEntity>()
        {
            new()
            {
                LeftId = 2,
                Name = "mary",
                Birth = new DateTime(2019, 1, 2),
                RightId = 0,
                CustomerId = 0,
                Addr = null
            },
        });
    }

    private List<MergeEntity> ThenGetResult()
    {
        var actual = _dbContext.QueryRawSql("SELECT * FROM M1")
            .Select(row => new MergeEntity
            {
                LeftId = (int) row["Customer_Id"],
                Name = (string) row["Name"],
                Birth = (DateTime) row["Birth"],
                RightId = GetRow<int>(row, "ExtraCustomer_Id"),
                CustomerId = GetRow<int>(row, "CustomerId"),
                Addr = GetRow<string>(row, "Addr"),
            })
            .ToList();
        return actual;
    }

    private T? GetRow<T>(Dictionary<string, object?> row, string name)
    {
        if (!row.ContainsKey(name))
        {
            return default;
        }

        var value = row[name];
        if (value == null)
        {
            return default;
        }
        return (T)value;
    }

    private void GiveData()
    {
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
            CustomerId = 3,
            Address = "Taichung"
        });
    }

    private void GiveTables()
    {
        var allTableNames = _dbContext.GetAllTableNames().Select(x => x.ToLower()).ToList();
        if (!allTableNames.Contains("customer"))
        {
            _dbContext.CreateTableByEntity(typeof(CustomerEntity));
        }

        if (!allTableNames.Contains("extraCustomer".ToLower()))
        {
            _dbContext.CreateTableByEntity(typeof(ExtraCustomerEntity));
        }
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
    [Column("Customer_Id")]
    public int LeftId { get; set; }
    public string Name { get; set; }
    public DateTime Birth { get; set; }
    
    [Column("ExtraCustomer_Id")]
    public int RightId { get; set; }
    public int CustomerId { get; set; }
    public string? Addr { get; set; }
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