using Microsoft.Data.Sqlite;
using Microsoft.EntityFrameworkCore;
using QueryKits.Services;

namespace QueryKitsTests;

public class SqliteMemoryDbContextOptionsFactory : IDbContextOptionsFactory
{
    public DbContextOptions<T> Create<T>() where T : DbContext
    {
        var cn = new SqliteConnection("data source=:memory:");
        cn.Open();
        return new DbContextOptionsBuilder<T>()
            .UseSqlite(cn)
            .Options;
    }
}