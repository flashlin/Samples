using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace PizzaWeb.Models
{

	public class PizzaDbContext : DbContext
	{
		private PizzaDbConfig _dbConfig;

		[ActivatorUtilitiesConstructor]
		public PizzaDbContext(IOptions<PizzaDbConfig> dbConfig)
		{
			_dbConfig = dbConfig.Value;
		}

		//public PizzaDbContext(IDbContextOptionsFactory optionsFactory) : base(optionsFactory.Create())
		//{
		//}
		public PizzaDbContext(DbContextOptions options) : base(options)
		{
		}

		public DbSet<StoreShelvesEntity> StoreShelves { get; set; }

		protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
		{
			optionsBuilder.UseSqlServer(_dbConfig.ConnectionString);
		}
	}
}
