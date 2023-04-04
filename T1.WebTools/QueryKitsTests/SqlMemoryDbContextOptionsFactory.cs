using Microsoft.EntityFrameworkCore;
using QueryKits.Services;
using T1.Standard.Data.SqlBuilders;

namespace QueryKitsTests;

/// <summary>
/// Not Recommended
/// </summary>
public class SqlMemoryDbContextOptionsFactory : IDbContextOptionsFactory
{
    public DbContextOptions<T> Create<T>() where T : DbContext
    {
        var inMemDbName = "D" + Guid.NewGuid();
        return new DbContextOptionsBuilder<T>()
            .UseInMemoryDatabase(inMemDbName)
            .Options;
    }

    public ISqlBuilder CreateSqlBuilder()
    {
        return new SqliteSqlBuilder();
    }
}