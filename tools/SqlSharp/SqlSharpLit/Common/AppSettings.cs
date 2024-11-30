using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Serilog;

namespace SqlSharpLit.Common;

public class AppSettings
{
    public IConfigurationRoot Load(string basePath)
    {
        var environment = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Development";
        var configuration = new ConfigurationBuilder()
            .SetBasePath(basePath)
            .AddJsonFile("appSettings.json", optional: false, reloadOnChange: true)
            .AddJsonFile($"appSettings.{environment}.json", optional: true, reloadOnChange: true)
            .Build();
        return configuration;
    }
}

public static class HostApplicationBuilderExtensions
{
    public static void LoadAppSettings(this HostApplicationBuilder builder)
    {
        var environment = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Development";
        builder.Configuration
            .SetBasePath(AppContext.BaseDirectory)
            .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
            .AddJsonFile($"appsettings.{environment}.json", optional: true);
    }

    public static void AddSerilog(this HostApplicationBuilder builder)
    {
        builder.Logging.AddSerilog(new LoggerConfiguration()
            .ReadFrom.Configuration(builder.Configuration)
            .CreateLogger());
    }
}