using ChatTrainDataWeb.Models.Entities;
using Microsoft.EntityFrameworkCore;

namespace ChatTrainDataWeb.Models.Repositories;

public class ChatTrainDataContext : DbContext
{
    public ChatTrainDataContext(IHostEnvironment hostEnvironment)
    {
        var path = hostEnvironment.ContentRootPath;
        DbPath = Path.Join(path, "ChatTrainData.db");
        Database.EnsureCreated();
    }

    public DbSet<TrainDataEntity> TrainData { get; set; } = null!;

    public string DbPath { get; }

    protected override void OnConfiguring(DbContextOptionsBuilder options)
    {
        options.UseSqlite($"Data Source={DbPath}");
    }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        // 配置 TrainData 表格
        modelBuilder.Entity<TrainDataEntity>(entity =>
        {
            entity.ToTable("TrainData");
            entity.HasKey(e => e.Id);
            entity.Property(x => x.Id)
                .ValueGeneratedOnAdd();
            // 添加其他属性和配置
            // entity.Property(e => e.PropertyName).IsRequired();
            // entity.Property(e => e.PropertyName).HasMaxLength(100);
            // ...
        });
    }
}