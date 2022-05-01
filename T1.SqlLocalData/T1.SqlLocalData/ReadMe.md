
首先建立 xUnit 測試專案, 使用 SqlLocalDb 物件建立實例
```
public class SqlLocalDbTest : IDisposable
{
    private string _instanceName = "local_db_instance";
    private string _databaseName = "Northwind";
    private string _databaseFile = @"D:\Demo\Northwind.mdf";
    private readonly SqlLocalDb _localDb = new SqlLocalDb();

    public SqlLocalDbTest()
    {
        _localDb.EnsureInstanceCreated(_instanceName);
        _localDb.ForceDropDatabase(_instanceName, _databaseName);
        _localDb.DeleteDatabaseFile(_databaseFile);
        _localDb.CreateDatabase(_instanceName, _databaseFile);
    }
}
```
EnsureInstanceCreated 方法建立為 local_db_instance 實例
ForceDropDatabase 強制刪除現有的資料庫
DeleteDatabaseFile 刪除 mdf 檔案
CreateDatabase 建立資料庫

將產品專案中的 connectionString 重指向 local_db_instance
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


如此一來就能夠在測試專案中, 直接執行資料庫整合測試
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