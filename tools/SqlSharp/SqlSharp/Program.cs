// See https://aka.ms/new-console-template for more information

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SqlSharp;
using SqlSharpLit.Common;
using SqlSharp.CommandPattern;
using SqlSharpLit;
using SqlSharpLit.Common.ParserLit;
using SqlSharpLit.Shared;

new AppSettings().LoadFile(AppContext.BaseDirectory);
var builder = Host.CreateApplicationBuilder(args);
builder.AddSerilog();

var services = builder.Services;
services.AddSqlSharpServices(builder.Configuration);
services.AddTransient<ExtractTableDataCommand>();
services.AddTransient<ExtractCreateTableSqlFromFolderCommand>();

var app = builder.Build();
var serviceProvider = app.Services;

var options = await LineCommandParseHelper.ParseAsync<SqlSharpOptions>(args);

var command = new CommandBuilder<SqlSharpOptions>()
     .Use(serviceProvider.GetRequiredService<ExtractTableDataCommand>())
     .Use(serviceProvider.GetRequiredService<ExtractCreateTableSqlFromFolderCommand>())
     .Build();
await command.ExecuteAsync(options);
     