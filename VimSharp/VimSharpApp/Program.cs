// See https://aka.ms/new-console-template for more information
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using VimSharpApp;

var builder = Host.CreateApplicationBuilder(args);

builder.Services.AddSingleton<Main>();

var host = builder.Build();

var main = host.Services.GetRequiredService<Main>();
main.Run();

