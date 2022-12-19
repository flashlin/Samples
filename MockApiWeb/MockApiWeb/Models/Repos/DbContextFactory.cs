using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;

namespace MockApiWeb.Models.Repos;

public class DbContextFactory
{
    public DbContextOptions<TContext> CreateMemoryOption<TContext>()
        where TContext: DbContext
    {
        var cn = new SqliteConnection("data source=:memory:");
        cn.Open();
        return new DbContextOptionsBuilder<TContext>()
            .UseSqlite(cn)
            .Options;
    }
    
    public TContext CreateFile<TContext>(string dbFile)
        where TContext: DbContext
    {
        var db = (TContext)Activator.CreateInstance(typeof(TContext), CreateFileOption<TContext>(dbFile))!;
        db.Database.EnsureCreated();
        return db;
    }
    
    public DbContextOptions<TContext> CreateFileOption<TContext>(string dbFile)
        where TContext: DbContext
    {
        var cn = new SqliteConnection($"data source={dbFile}");
        cn.Open();
        return new DbContextOptionsBuilder<TContext>()
            .UseSqlite(cn)
            .Options;
    }
}