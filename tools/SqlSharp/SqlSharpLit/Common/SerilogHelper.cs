using Microsoft.Extensions.Hosting;
using Serilog;

namespace SqlSharpLit.Common;

public static class SerilogHelper
{
    public static void AddSerilog(this IHostApplicationBuilder builder)
    {
        builder.Logging.AddSerilog(new LoggerConfiguration()
            .ReadFrom.Configuration(builder.Configuration)
            .Enrich.FromLogContext()
            .CreateLogger());
    }
}