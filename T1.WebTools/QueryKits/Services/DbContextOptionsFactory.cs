using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using T1.Standard.Data.SqlBuilders;

namespace QueryKits.Services;

public class DbContextOptionsFactory : IDbContextOptionsFactory
{
    private readonly DbConfig _dbConfig;

    public DbContextOptionsFactory(IOptions<DbConfig> dbConfig)
    {
        _dbConfig = dbConfig.Value;
    }

    public DbContextOptions<T> Create<T>()
        where T : DbContext
    {
        return new DbContextOptionsBuilder<T>()
            .UseSqlServer(_dbConfig.ConnectionString)
            .Options;
    }

    public ISqlBuilder CreateSqlBuilder()
    {
        return new MssqlBuilder();
    }
}