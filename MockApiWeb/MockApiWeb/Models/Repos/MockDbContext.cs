using Microsoft.EntityFrameworkCore;
using MockApiWeb.Controllers;

namespace MockApiWeb.Models.Repos;

public class MockDbContext : DbContext
{
    public MockDbContext(DbContextOptions<MockDbContext> options)
        : base(options)
    {
    }

    public DbSet<WebApiFuncInfoEntity> WebApiFuncInfos { get; set; } = null!;
}