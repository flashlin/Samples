// See https://aka.ms/new-console-template for more information

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SqlSharp;
using SqlSharpLit.Common;
using SqlSharp.CommandPattern;
using SqlSharpLit;
using SqlSharpLit.Shared;

new AppSettings().LoadFile(AppContext.BaseDirectory);
var builder = Host.CreateApplicationBuilder(args);
builder.AddSerilog();
var services = builder.Services;
services.AddSqlSharpServices(builder.Configuration);

var options = await LineCommandParseHelper.ParseAsync<SqlSharpOptions>(args);

// var command = new CommandBuilder<SqlSharpOptions>()
//     .Use(new ExtractTableDataCommand())
//     .Build();