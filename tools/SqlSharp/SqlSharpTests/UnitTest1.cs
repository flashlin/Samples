using FluentAssertions;
using Microsoft.EntityFrameworkCore;
using SqlSharpLit;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Hosting;

namespace SqlSharpTests;

public class Tests
{
    private IHost _host;
    private IServiceProvider _serviceProvider;
    private IConfiguration _configuration;
    private DynamicDbContext _db;

    [SetUp]
    public void Setup()
    {
        var environment = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Development";
        _configuration = new ConfigurationBuilder()
            .SetBasePath(TestContext.CurrentContext.TestDirectory)
            .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
            .AddJsonFile($"appsettings.{environment}.json", optional: true, reloadOnChange: true)
            .Build();
        
        var builder = Host.CreateApplicationBuilder();
        var services = builder.Services;
        
        services.AddSingleton(_configuration);
        services.Configure<DbConfig>(_configuration.GetSection("ConnectionStrings"));
        services.AddDbContextPool<DynamicDbContext>(options => options.UseSqlServer(_configuration.GetConnectionString("DbServer")));
        
        _host = builder.Build();
        _serviceProvider = _host.Services;

        _db = _serviceProvider.GetRequiredService<DynamicDbContext>();
        _db.Database.ExecuteSql($"""
                                 INSERT INTO [dbo].[Customer] ([Name], [Email]) VALUES ('John Doe', 'test1@mail.com')
                                 INSERT INTO [dbo].[Customer] ([Name], [Email]) VALUES ('Mary', 'test2@mail.com')
                                 """);
    }
    
    [Test]
    public void Test1()
    {
        var fields = _db.GetTableSchema("Customer");
        var data = _db.GetTopNTableData(1, "Customer", fields, null);
        data.Should().BeEquivalentTo([
            new Dictionary<string, string>()
            {
                ["Id"] = "1",
                ["Name"] = "John Doe",
                ["Email"] = "test1@mail.com"
            }
        ]);
    }
}