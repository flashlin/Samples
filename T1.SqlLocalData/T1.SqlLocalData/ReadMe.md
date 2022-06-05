
First, Create xUnit or other test project, 
and new SqlLocalDb to create instance

```
using T1.SqlLocalData;

public class SqlLocalDbTest : IDisposable
{
    private string _instanceName = "local_db_instance";
    private string _databaseName = "Northwind";
    private readonly SqlLocalDb _localDb = new SqlLocalDb(@"D:\Demo");

    public SqlLocalDbTest()
    {
        _localDb.EnsureInstanceCreated(_instanceName);
        _localDb.ForceDropDatabase(_instanceName, _databaseName);
        _localDb.DeleteDatabaseFile(_databaseName);
        _localDb.CreateDatabase(_instanceName, _databaseName);
    }
}
```

* EnsureInstanceCreated: Ensure local_db_instance instance created
* ForceDropDatabase:  Force delete exists database
* DeleteDatabaseFile: Delete mdf ldf files
* CreateDatabase: Create database

<br/>
<br/>

The SqlLocalDb.exe default installed location is
"C:\Program Files\Microsoft SQL Server\150\Tools\Binn".
If you installed it in other location, 
then you can change default location by SetInstalledLocation method.
```
_localDb.SetInstalledLocation("D:\\OtherLocation\\Binn");
```


<br/>

Setting connectionString to local_db_instance
```
public class MyDbContext : DbContext
{
	string _connectionString = "Server=(localdb)\\local_db_instance;Integrated security=SSPI;database=Northwind;";
	
	public DbSet<CustomerEntity> Customers { get; set; }
	
	protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
	{
		optionsBuilder.UseSqlServer(_connectionString);
	}
}
```

<br/>
Finish, you can use local_db_instance to test your code in Integrated Database Test Project.

```
[Fact]
public void execute_store_procedure()
{
    var myDb = new MyDbContext();
    myDb.Database.ExecuteSqlRaw(@"CREATE TABLE customer (id INT PRIMARY KEY, name VARCHAR(50))");
    myDb.Database.ExecuteSqlRaw(@"INSERT customer(id,name) VALUES (1,'Flash'),(3,'Jack'),(4,'Mary')");
    myDb.Database.ExecuteSqlRaw(@"CREATE PROC MyGetCustomer 
        @id INT AS 
        BEGIN 
            SET NOCOUNT ON; 
            select name from customer 
            WHERE id=@id 
        END");

    var customer = myDb.QuerySqlRaw<CustomerEntity>("EXEC MyGetCustomer @id", new
    {
        id = 3
    }).First();

    Assert.Equal("Jack",customer.Name);
}
```


