﻿using GitCli.Models;
using GitCli.Models.Repositories;
using Microsoft.Extensions.Hosting;
using T1.ConsoleUiMixedReality;

var hostBuilder = new HostFactory().Create(args);

var host = hostBuilder
	.ConfigureServices(services =>
	{
		 services.AddDbContext<GitCliDbContext>();
		 services.AddTransient<IGitRepoAgent, GitRepoAgent>();
		 services.AddTransient<IConsoleWriter, ConsoleWriter>();
		 services.AddSingleton<ConsoleManager>();
		 
		 
		 //services.AddTransient<IConsoleWriter, ConsoleWriter>();
		 //services.AddSingleton<ITerminalGui, TerminalGui>();
		 services.AddTransient<GitStatusCommand>();
		 services.AddSingleton<IConsoleWindow, ConsoleWindow>();
		 services.AddSingleton<IApplicationWindow>(sp => sp.GetRequiredService<IConsoleWindow>());
	})
	.Build();

var main = host.Services.GetService<IApplicationWindow>();
await main!.Run(args);
