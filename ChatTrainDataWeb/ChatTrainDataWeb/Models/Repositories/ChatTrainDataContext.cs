using ChatTrainDataWeb.Models.Entities;
using Microsoft.EntityFrameworkCore;

namespace ChatTrainDataWeb.Models.Repositories;

public class ChatTrainDataContext : DbContext
{
    public DbSet<TrainDataEntity> TrainData { get; set; }

    public string DbPath { get; }

    public ChatTrainDataContext()
    {
        var folder = Environment.SpecialFolder.LocalApplicationData;
        var path = Environment.GetFolderPath(folder);
        DbPath = System.IO.Path.Join(path, "blogging.db");
    }

    protected override void OnConfiguring(DbContextOptionsBuilder options)
        => options.UseSqlite($"Data Source={DbPath}");
}