using CloneSqlServer.Kit;
using FluentAssertions;

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
                                   IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'test')
                                   BEGIN
                                       CREATE DATABASE [test];
                                   END
                                   """);
    }

    [TearDown]
    public async Task TearDown()
    {
        await _localDb.DisposeAsync();
    }

    [Test]
    public async Task TableSchema()
    {
        var tableSchemaList = await _localDb.QueryTableSchemaAsync();
        tableSchemaList.Should().BeEquivalentTo(
            new List<TableSchema>());
    }
}