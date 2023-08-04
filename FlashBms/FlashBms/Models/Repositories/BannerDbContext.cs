using FlashBms.Models.Entities;
using Microsoft.EntityFrameworkCore;

namespace FlashBms.Models.Repositories;

public class BannerDbContext : DbContext
{
    public BannerDbContext(IHostEnvironment hostEnvironment)
    {
        var path = Path.Combine(hostEnvironment.ContentRootPath, "Data");
        DbPath = Path.Join(path, "Banner.db");
        Database.EnsureCreated();
    }

    public DbSet<BannerTemplateEntity> BannerTemplates { get; set; } = null!;

    public string DbPath { get; }

    protected override void OnConfiguring(DbContextOptionsBuilder options)
    {
        options.UseSqlite($"Data Source={DbPath}");
    }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        // 配置 TrainData 表格
        modelBuilder.Entity<BannerTemplateEntity>(entity =>
        {
            entity.ToTable("BannerTemplate");
            entity.HasKey(e => e.Id);
            entity.Property(x => x.Id)
                .ValueGeneratedOnAdd();
            // entity.Property(e => e.PropertyName).IsRequired();
            // entity.Property(e => e.PropertyName).HasMaxLength(100);
        });
    }
}