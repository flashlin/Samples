using System.Text;
using System.Text.Json;
using FluentAssertions;
using Microsoft.EntityFrameworkCore;
using SqlSharpLit;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using SqlSharpLit.Common;
using SqlSharpLit.Shared;
using T1.Standard.Serialization;
using JsonSerializer = System.Text.Json.JsonSerializer;

namespace SqlSharpTests;

[TestFixture]
public class DynamicDbContextTests
{
    private IConfiguration _configuration;
    private DynamicDbContext _db;
    private IHost _host;
    private ILogger<DynamicDbContextTests> _logger;
    private IServiceProvider _serviceProvider;

    [Test]
    public void GetCustomerTableSchema()
    {
        var fields = _db.GetTableSchema("Customer");
        fields.Should().BeEquivalentTo([
            new TableSchemaEntity
            {
                Name = "Id",
                DataType = "int",
                IsNull = false,
                IsPk = true
            },
            new TableSchemaEntity
            {
                Name = "Name",
                DataType = "nvarchar",
                IsNull = false,
                IsPk = false
            },
            new TableSchemaEntity
            {
                Name = "Email",
                DataType = "varchar",
                IsNull = false,
                IsPk = false
            },
        ]);
    }

    [Test]
    public void GetTopNTableData()
    {
        var fields = _db.GetTableSchema("Customer");

        var data = _db.GetTopNTableData(1, "Customer", fields, null);
        _logger.LogInformation("GetTopNTableData: {data}", Serialize(data));
        data.Should().BeEquivalentTo([
            new Dictionary<string, string>()
            {
                ["Id"] = "1",
                ["Name"] = "John Doe",
                ["Email"] = "test1@mail.com"
            }
        ]);
    }

    [Test]
    public void GetTopNTableData2()
    {
        var fields = _db.GetTableSchema("Customer");

        var data = _db.GetTopNTableData(1, "Customer", fields, null);
        _logger.LogInformation("data1: {data}", Serialize(data));
        data.Should().BeEquivalentTo([
            new Dictionary<string, string>()
            {
                ["Id"] = "1",
                ["Name"] = "John Doe",
                ["Email"] = "test1@mail.com"
            }
        ]);
        
        var nextAccumulator = data.Last()["Id"];
        data = _db.GetTopNTableData(1, "Customer", fields, nextAccumulator);
        _logger.LogInformation("data2: {data}", Serialize(data)); 
        data.Should().BeEquivalentTo([
            new Dictionary<string, string>()
            {
                ["Id"] = "2",
                ["Name"] = "Mary",
                ["Email"] = "test2@mail.com"
            }
        ]);
    }

    [Test]
    public void ExportTableData()
    {
        var data = _db.ExportTableData("Customer");
        data.Should().BeEquivalentTo([
            new Dictionary<string, string>()
            {
                ["Id"] = "1",
                ["Name"] = "John Doe",
                ["Email"] = "test1@mail.com"
            },
            new Dictionary<string, string>()
            {
                ["Id"] = "2",
                ["Name"] = "Mary",
                ["Email"] = "test2@mail.com"
            }
        ]);
    }

    private string Serialize(Dictionary<string, string> dict)
    {
        var options = new JsonSerializerOptions();
        options.Converters.Add(new DictionaryJsonConverter<string,string>());
        return JsonSerializer.Serialize(dict, options);
    }

    private string Serialize(IEnumerable<Dictionary<string, string>> dictList)
    {
        var sb = new StringBuilder();
        foreach (var item in dictList)
        {
            sb.AppendLine(Serialize(item));
        }

        return sb.ToString();
    }

    [OneTimeSetUp]
    public void Setup()
    {
        var loggerFactory = LoggerFactory.Create(builder => 
        {
            builder.AddConsole()
                .AddDebug(); 
        });
        _logger = loggerFactory.CreateLogger<DynamicDbContextTests>();

        new AppSettings().LoadFile(TestContext.CurrentContext.TestDirectory);
        var builder = Host.CreateApplicationBuilder();
        var services = builder.Services;
        services.AddSqlSharpServices(_configuration);
        
        _host = builder.Build();
        _serviceProvider = _host.Services;

        _db = _serviceProvider.GetRequiredService<DynamicDbContext>();
        _db.Database.ExecuteSql($"""DELETE FROM [dbo].[Customer]""");
        _db.Database.ExecuteSql($"""
                                 INSERT INTO [dbo].[Customer] ([Name], [Email]) VALUES ('John Doe', 'test1@mail.com')
                                 INSERT INTO [dbo].[Customer] ([Name], [Email]) VALUES ('Mary', 'test2@mail.com')
                                 """);
    }
}