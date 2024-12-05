// See https://aka.ms/new-console-template for more information

using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Serilog;
using SqlSharp;
using SqlSharpLit.Common;
using SqlSharp.CommandPattern;
using SqlSharp.Commands;
using SqlSharpLit.Shared;

var builder = Host.CreateApplicationBuilder(args);
builder.LoadAppSettings();
builder.AddSerilog();


var services = builder.Services;
services.AddSqlSharpServices(builder.Configuration);
services.AddTransient<ExtractTableDataCommand>();
services.AddTransient<ExtractCreateTableSqlFromFolderCommand>();
services.AddTransient<ExtractSelectSqlFromFolderCommand>();

var app = builder.Build();
var serviceProvider = app.Services;

var options = await LineCommandParseHelper.ParseAsync<SqlSharpOptions>(args);

var command = new SpecificationAsyncEvaluator<SqlSharpOptions, Task>([
     serviceProvider.GetRequiredService<ExtractTableDataCommand>(),
     serviceProvider.GetRequiredService<ExtractCreateTableSqlFromFolderCommand>(),
     serviceProvider.GetRequiredService<ExtractSelectSqlFromFolderCommand>()
]);
await command.EvaluateAsync(options);
     