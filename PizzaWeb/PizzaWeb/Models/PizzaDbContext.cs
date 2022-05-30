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

		public DbSet<StoreShelvesEntity> StoreShelves => Set<StoreShelvesEntity>();
		public DbSet<BannerTemplateEntity> BannerTemplates => Set<BannerTemplateEntity>();
		public DbSet<BannerEntity> Banners => Set<BannerEntity>();
		public DbSet<BannerResxEntity> BannerResx => Set<BannerResxEntity>();
	}
}
