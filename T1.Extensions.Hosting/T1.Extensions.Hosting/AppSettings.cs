using Microsoft.Extensions.Configuration;

namespace T1.Extensions.Hosting;

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