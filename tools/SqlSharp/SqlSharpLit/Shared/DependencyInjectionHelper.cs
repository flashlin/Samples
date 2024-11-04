using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace SqlSharpLit.Shared;

public static class DependencyInjectionHelper
{
    public static void AddSqlSharpServices(this IServiceCollection services,IConfiguration configuration)
    {
        services.AddSingleton(configuration);
        services.Configure<DbConfig>(configuration.GetSection("ConnectionStrings"));
        services.AddDbContextPool<DynamicDbContext>(options => options.UseSqlServer(configuration.GetConnectionString("DbServer")));
    }
}