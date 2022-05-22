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

		public void SetDbConnectionString(string connectionString)
		{
			_dbConfig = new PizzaDbConfig
			{
				ConnectionString = connectionString
			};
		}

		public DbContextOptions Create()
		{
			return new DbContextOptionsBuilder<PizzaDbContext>()
				.UseSqlServer(_dbConfig.ConnectionString)
				.Options;
		}
	}

	public class UseSqlServerByConnectionString : IDbContextOptionsFactory
	{
		private string _connectionString;

		public UseSqlServerByConnectionString(string connectionString)
		{
			_connectionString = connectionString;
		}

		public DbContextOptions Create()
		{
			return new DbContextOptionsBuilder<PizzaDbContext>()
				.UseSqlServer(_connectionString)
				.Options;
		}
	}
}
