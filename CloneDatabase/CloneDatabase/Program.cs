using System.Security.Authentication.ExtendedProtection;
using CloneDatabase;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

static IHostBuilder CreateHostBuilder(string[] args) =>
    Host.CreateDefaultBuilder(args)
        .ConfigureHostConfiguration(configuration =>
        {
            configuration.AddJsonFile("appSettings.json", optional: false, reloadOnChange: true);
        })
        .ConfigureServices((context, services) =>
        {
            services.Configure<DbConfig>(context.Configuration);
            services.AddTransient<DatabaseCloner>();
            services.AddTransient<ISqlScriptBuilder, MsSqlScriptBuilder>();
        });


var host = CreateHostBuilder(args).Build();

var helper = host.Services.GetRequiredService<DatabaseCloner>();
helper.Clone();
Console.WriteLine("=== END ===");