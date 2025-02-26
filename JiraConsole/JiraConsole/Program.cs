// See https://aka.ms/new-console-template for more information

using JiraConsole;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

DotNetEnv.Env.Load();

var configuration = new ConfigurationBuilder()
    .AddEnvironmentVariables()
    .Build();

var services = new ServiceCollection();
services.AddHttpClient();
services.Configure<JiraConfig>(config =>
{
    config.BaseUrl = "https://ironman.atlassian.net/wiki/spaces/KM";
    config.UserName = configuration["JIRA_NAME"]!;
    config.ApiKey = configuration["JIRA_APIKEY"]!;
});
services.AddTransient<JiraHelper>();


var serviceProvider = services.BuildServiceProvider();
var jiraHelper = serviceProvider.GetRequiredService<JiraHelper>();
await jiraHelper.Get();
Console.WriteLine("Hello, World!");