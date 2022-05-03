using System.Reflection;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.FileProviders;

namespace SqliteCli.Factories;

public interface IEnvironment
{
    string? GetEnvironmentVariable(string name);
}

public class RuntimeEnvironment : IEnvironment
{
    public string? GetEnvironmentVariable(string name)
    {
        return Environment.GetEnvironmentVariable(name);
    }
}

public static class MinimalWebAppExtension
{
    public static void StartAsync(this WebApplication webApp, Assembly assemblyCaller,
        IEnvironment environment,
        string contentFolderName = "Contents")
    {
        var port = environment.GetEnvironmentVariableOrDefault("port", "3000");
        webApp.Urls.Add($"http://127.0.0.1:{port}");
        webApp.UseDefaultFiles();
        webApp.UseStaticFiles(new StaticFileOptions
        {
            RequestPath = "/Embedded",
            FileProvider = new Microsoft.Extensions.FileProviders
                .ManifestEmbeddedFileProvider(assemblyCaller, "Contents"),
        });
        var appRootPath = Path.GetDirectoryName(assemblyCaller.Location);
        webApp.UseStaticFiles(new StaticFileOptions
        {
            RequestPath = "/Contents",
            FileProvider = new PhysicalFileProvider(
                Path.Combine(appRootPath, contentFolderName)),
        });
        webApp.UseRouting();
        webApp.MapControllerRoute(
            name: "default",
            pattern: "{controller=Home}/{action=Index}/{id?}");
        webApp.RunAsync();
    }
}