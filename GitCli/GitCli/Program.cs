using GitCli.Models;
using GitCli.Models.Repositories;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var hostBuilder = new HostFactory().Create(args);

var host = hostBuilder
	.ConfigureServices(services =>
	{
		 services.AddDbContext<GitCliDbContext>();
		 services.AddTransient<IGitRepoAgent, GitRepoAgent>();
		 services.AddSingleton<IApplicationWindow, ApplicationWindow>();
		 services.AddSingleton<Main>();
	})
	.Build();

var main = host.Services.GetService<Main>();
await main!.Run();
