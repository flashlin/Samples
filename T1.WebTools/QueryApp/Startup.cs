using System.Net;
using System.Net.Sockets;
using System.Reflection;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using QueryApp.Controllers.Apis;
using QueryApp.Models;
using QueryApp.Models.Clients;
using QueryApp.Models.Services;

namespace QueryApp;

public class Startup
{
    public void Run(string[] args)
    {
        var appUid = Guid.NewGuid().ToString();
        var appLocation = Path.GetDirectoryName(Assembly.GetEntryAssembly()!.Location)!;
        var port = FindAvailablePort();

        var hostBuilder = Host.CreateDefaultBuilder(args)
            .ConfigureServices(services =>
            {
                services.AddSingleton<ILocalEnvironment>(sp => new LocalEnvironment
                {
                    AppUid = appUid,
                    AppLocation = appLocation,
                    Port = port,
                });
                services.AddHostedService<EchoBackgroundService>();
            })
            .ConfigureWebHostDefaults(webBuilder =>
            {
                webBuilder.UseUrls($"http://localhost:{port}");
                webBuilder.UseStartup<Startup>();
            });

        var host = hostBuilder.Build();
        host.Run();
    }

    public void ConfigureServices(IServiceCollection services)
    {
        services.AddControllers();
        services.AddSingleton<ILocalDbService, LocalDbService>();
        services.AddTransient<ILocalQueryClient, LocalQueryClient>();
        services.AddTransient<IReportRepo, ReportDbContext>();
        services.AddSwaggerGen();
    }

    public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
    {
        app.UseRouting();
        app.UseSwagger();
        app.UseSwaggerUI();
        app.UseEndpoints(endpoints =>
        {
            endpoints.MapControllerRoute(
                name: "default",
                pattern: "{controller=Home}/{action=Index}/{id?}");
            endpoints.MapControllers();
        });
    }

    int FindAvailablePort()
    {
        TcpListener listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint) listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }
}