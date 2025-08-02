using CloneSqlServer.Kit;

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
                                   CREATE DATABASE [test];
                                   """);
    }

    [Test]
    public void Test1()
    {
        Assert.Pass();
    }
}