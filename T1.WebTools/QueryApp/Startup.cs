﻿using System.Net;
using System.Net.Sockets;
using System.Reflection;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Razor;
using Microsoft.AspNetCore.Mvc.ViewFeatures;
using Microsoft.AspNetCore.Mvc.ViewFeatures.Infrastructure;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using QueryApp.Controllers.Apis;
using QueryApp.Models;
using QueryApp.Models.Clients;
using QueryApp.Models.Services;
using QueryKits.Services;
using Serilog;
using Serilog.Extensions.Logging.File;
using T1.WebTools.CsvEx;

namespace QueryApp;

public class Startup
{
    public void Run(string[] args)
    {
        var appEnvironment = LocalEnvironment.Load();
        Console.WriteLine($"Starting '{appEnvironment.AppLocation}'");

        File.WriteAllText(appEnvironment.AppLocation + "/AppUid.txt", "[{" + $"""appUid:"{appEnvironment.AppUid}",port:{appEnvironment.Port}""" + "}]");
        //FileHelper.EnsureDirectory(Path.Combine(appLocation, "Logs"));

        var hostBuilder = Host.CreateDefaultBuilder(args)
            .ConfigureAppConfiguration((context, config) =>
            {
                config.AddJsonFile("appSettings.json", optional: true);
            })
            .ConfigureLogging(logging =>
            {
                logging.ClearProviders();
                //logging.AddConsole();
                logging.AddSerilog();
            })
            .UseSerilog((hostingContext, loggerConfiguration) =>
            {
                loggerConfiguration.ReadFrom.Configuration(hostingContext.Configuration);
            })
            .ConfigureServices(services =>
            {
                services.AddSingleton<ILocalEnvironment>(sp => appEnvironment);
                services.AddHostedService<EchoBackgroundService>();
            })
            .ConfigureWebHostDefaults(webBuilder =>
            {
                webBuilder.UseUrls($"http://localhost:{appEnvironment.Port}");
                webBuilder.UseStartup<Startup>();
                //webBuilder.UseWebRoot(Path.Combine(appLocation, "wwwroot"));
                webBuilder.UseStaticWebAssets();
            });

        var host = hostBuilder.Build();
        host.Run();
    }

    public void ConfigureServices(IServiceCollection services)
    {
        // services.AddMvcCore()
        //     .AddRazorRuntimeCompilation();
        services.AddSignalR();
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
        app.UseStaticFiles();
    }
}