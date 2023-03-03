using System.Net;
using System.Net.Sockets;
using System.Reflection;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Razor;
using Microsoft.AspNetCore.Mvc.ViewFeatures;
using Microsoft.AspNetCore.Mvc.ViewFeatures.Infrastructure;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using QueryApp.Controllers.Apis;
using QueryApp.Models;
using QueryApp.Models.Clients;
using QueryApp.Models.Services;
using Serilog;
using Serilog.Extensions.Logging.File;
using T1.WebTools.CsvEx;

namespace QueryApp;

public class Startup
{
    public void Run(string[] args)
    {
        var entryAssembly = Assembly.GetEntryAssembly()!;
        var appUid = Guid.NewGuid().ToString();
        var appLocation = Path.GetDirectoryName(entryAssembly.Location)!;
        var appVersion = entryAssembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()!
            .InformationalVersion;
        var port = FindAvailablePort();

        File.WriteAllText(appLocation + "/AppUid.txt", "[{" + $"""appUid:"{appUid}",port:{port}""" + "}]");

        var hostBuilder = Host.CreateDefaultBuilder(args)
            .UseSerilog((hostingContext, loggerConfiguration) => 
            {
                loggerConfiguration.ReadFrom.Configuration(hostingContext.Configuration);
            })
            .ConfigureServices(services =>
            {
                services.AddSingleton<ILocalEnvironment>(sp => new LocalEnvironment
                {
                    AppUid = appUid,
                    AppVersion = appVersion,
                    AppLocation = appLocation,
                    Port = port,
                });
                services.AddHostedService<EchoBackgroundService>();
            })
            .ConfigureWebHostDefaults(webBuilder =>
            {
                webBuilder.UseUrls($"http://localhost:{port}");
                webBuilder.UseStartup<Startup>();
                //webBuilder.UseWebRoot(Path.Combine(appLocation, "wwwroot"));
                //webBuilder.UseStaticWebAssets();
            });

        var host = hostBuilder.Build();
        host.Run();
    }

    public void ConfigureServices(IServiceCollection services)
    {
        // services.AddMvcCore()
        //     .AddRazorRuntimeCompilation();
        services.AddHttpClient();
        services.AddControllersWithViews()
            .AddRazorRuntimeCompilation();
        services.AddServerSideBlazor();
        services.AddRazorPages();
        // .AddJsonOptions(options =>
        // {
        //     options.JsonSerializerOptions.Converters.Add(new DictionaryStringToStringConverter());
        // });
        //services.AddServerSideBlazor();
        services.AddSingleton<ILocalDbService, LocalDbService>();
        services.AddTransient<ILocalQueryHostClient, LocalQueryHostClient>();
        services.AddTransient<IReportRepo, ReportDbContext>();
        services.AddTransient<IMyJsonSerializer, MyJsonSerializer>();
        services.AddSwaggerGen();
        services.AddCors(options =>
        {
            options.AddPolicy("AllowAll",
                cp => cp.AllowAnyOrigin()
                    .AllowAnyMethod()
                    .AllowAnyHeader());
        });
    }

    public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
    {
        app.UseCors("AllowAll");
        app.UseRouting();
        app.UseSwagger();
        app.UseSwaggerUI();
        app.UseEndpoints(endpoints =>
        {
            endpoints.MapControllerRoute(
                name: "default",
                pattern: "{controller=Home}/{action=Index}/{id?}");
            endpoints.MapControllers();
            endpoints.MapBlazorHub();
            endpoints.MapRazorPages();
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