using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using PizzaWeb.Models.Banner;

namespace PizzaWeb.Models
{

	public class PizzaDbContext : DbContext
	{
		public PizzaDbContext(DbContextOptions options)
		: base(options)
		{
		}

		public DbSet<StoreShelvesEntity> StoreShelves { get; set; }
		public DbSet<BannerTemplateEntity> BannerTemplates { get; set; }
		public DbSet<BannerVariableEntity> BannerVariables { get; set; }
		public DbSet<BannerResxEntity> BannerResx { get; set; }
	}
}
