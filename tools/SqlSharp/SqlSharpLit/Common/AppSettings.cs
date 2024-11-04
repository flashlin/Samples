using Microsoft.Extensions.Configuration;
namespace SqlSharpLit.Common;

public class AppSettings
{
    public IConfigurationRoot LoadFile(string basePath)
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