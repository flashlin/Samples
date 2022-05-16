using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace PizzaWeb.Models
{
	public class SqlServerDbContextOptionsFactory : IDbContextOptionsFactory
	{
		private PizzaDbConfig _dbConfig;

		public SqlServerDbContextOptionsFactory(IOptions<PizzaDbConfig> dbConfig)
		{
			_dbConfig = dbConfig.Value;
		}

		public DbContextOptions Create()
		{
			return new DbContextOptionsBuilder<PizzaDbContext>()
				.UseSqlServer(_dbConfig.ConnectionString)
				.Options;
		}
	}
}
