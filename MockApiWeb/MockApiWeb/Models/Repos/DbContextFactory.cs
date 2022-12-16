using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;

namespace MockApiWeb.Models.Repos;

public class DbContextFactory
{
    public TContext Create<TContext>()
        where TContext: DbContext
    {
        var cn = new SqliteConnection("data source=:memory:");
        cn.Open();
        var opt = new DbContextOptionsBuilder<TContext>()
            .UseSqlite(cn).Options;
        var db = (TContext)Activator.CreateInstance(typeof(TContext), opt)!;
        db.Database.EnsureCreated();
        return db;
    }
    
    public TContext CreateFile<TContext>(string dbFile)
        where TContext: DbContext
    {
        var cn = new SqliteConnection($"data source={dbFile}");
        cn.Open();
        var opt = new DbContextOptionsBuilder<TContext>()
            .UseSqlite(cn).Options;
        var db = (TContext)Activator.CreateInstance(typeof(TContext), opt)!;
        db.Database.EnsureCreated();
        return db;
    }
}