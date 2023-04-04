using Microsoft.EntityFrameworkCore;
using T1.Standard.Data.SqlBuilders;

namespace QueryKits.Services;

public interface IDbContextOptionsFactory
{
    DbContextOptions<T> Create<T>()
        where T : DbContext;

    ISqlBuilder CreateSqlBuilder();
}