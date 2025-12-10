using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace T1.SqlSharpTests;

public class TestDbContext : DbContext
{
    public DbSet<UserEntity> Users { get; set; }

    public TestDbContext(DbContextOptions<TestDbContext> options)
        : base(options)
    {
    }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<UserEntity>(ConfigureUserEntity);
    }

    private void ConfigureUserEntity(EntityTypeBuilder<UserEntity> entity)
    {
        entity.ToTable("Users", "dbo");
        entity.HasKey(e => e.Id);
        entity.Property(e => e.Id).ValueGeneratedOnAdd();
    }
}
