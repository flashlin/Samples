using Microsoft.EntityFrameworkCore;
using T1.EfCore;

namespace T1.EFCoreTests;

public class Tests
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void Test1()
    {
        var db = new MyDbContext();
        
        db.Database.ExecuteSqlRaw($"""
                                   CREATE TABLE [dbo].[Customer] (
                                     [Id] [int] NOT NULL,
                                     [Name] [nvarchar](50) NOT NULL,
                                     CONSTRAINT [PK_Customer] PRIMARY KEY ([Id])
                                   )
                                   """);
        
        db.Upsert(new MyEntity
        {
            Id = 1,
            Name = "flash"
        }).On(x => x.Id)
            .Execute();
        
        
    }
}

public class MyDbContext : DbContext
{
    public DbSet<MyEntity> Customer { get; set; }
    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
    {
        optionsBuilder.UseSqlServer("Server=(localdb)\\MSSQLLocalDB;Database=NorthWind;Trusted_Connection=True;");
    }
    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<MyEntity>(entity =>
        {
            entity.Property(e => e.Name)
                .HasColumnType("nvarchar(50)")
                .IsRequired();
        });
    }
}

public class MyEntity
{
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
}