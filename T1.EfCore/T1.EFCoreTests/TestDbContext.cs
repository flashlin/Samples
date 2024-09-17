using Microsoft.EntityFrameworkCore;

namespace T1.EFCoreTests;

public class TestDbContext : DbContext
{
    public DbSet<CustomerEntity> Customer { get; set; }
    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        optionsBuilder.UseSqlServer("Server=(localdb)\\MSSQLLocalDB;Database=NorthWind;Trusted_Connection=True;");
    }
    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<CustomerEntity>(entity =>
        {
            entity.Property(e => e.Name)
                .HasColumnType("nvarchar(50)")
                .IsRequired();
        });
    }
}