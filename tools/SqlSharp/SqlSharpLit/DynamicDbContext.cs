using Microsoft.EntityFrameworkCore;

namespace SqlSharpLit;

public class DynamicDbContext : DbContext
{
    public DynamicDbContext(DbContextOptions<DynamicDbContext>? options) 
        : base(options ?? CreateDbContextOptions(null)) 
    {
    }

    public static DbContextOptions<DynamicDbContext> CreateDbContextOptions(string? connectionString)
    {
        connectionString ??= @".\\SQLExpress;Integrated Security=true;";
        var options = new DbContextOptionsBuilder<DynamicDbContext>()
            .UseSqlServer(connectionString)
            .Options;
        return options;
    }
}