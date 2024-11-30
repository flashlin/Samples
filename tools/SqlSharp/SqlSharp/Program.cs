// See https://aka.ms/new-console-template for more information

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SqlSharp;
using SqlSharpLit.Common;
using SqlSharp.CommandPattern;
using SqlSharpLit.Shared;

new AppSettings().Load(AppContext.BaseDirectory);
var builder = Host.CreateApplicationBuilder(args);
builder.AddSerilog();

var services = builder.Services;
services.AddSqlSharpServices(builder.Configuration);
services.AddTransient<ExtractTableDataSpecification>();
services.AddTransient<ExtractCreateTableSqlFromFolderSpecification>();

var app = builder.Build();
var serviceProvider = app.Services;

var options = await LineCommandParseHelper.ParseAsync<SqlSharpOptions>(args);

var command = new SpecificationEvaluator<SqlSharpOptions, Task>([
     serviceProvider.GetRequiredService<ExtractTableDataSpecification>(),
     serviceProvider.GetRequiredService<ExtractCreateTableSqlFromFolderSpecification>(),
]);
await command.EvaluateAsync(options);
     