using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using PizzaWeb.Models.Banner;

namespace PizzaWeb.Models
{

	public class PizzaDbContext : DbContext
	{
		//public PizzaDbContext(IDbContextOptionsFactory optionsFactory) 
		//	: base(optionsFactory.Create())
		//{
		//}

		public PizzaDbContext(DbContextOptions options)
		: base(options)
		{
		}

		public DbSet<StoreShelvesEntity> StoreShelves { get; set; }
		public DbSet<BannerTemplateEntity> BannerTemplates { get; set; }
	}

	public class RepositoryFactory : IRepositoryFactory
	{
		private readonly IServiceProvider _serviceProvider;

		public RepositoryFactory(IServiceProvider serviceProvider)
		{
			_serviceProvider = serviceProvider;
		}

		public T BuildRepository<T>()
        where T : DbContext
		{
			return _serviceProvider.GetService<IDbContextFactory<T>>()
				.CreateDbContext();
		}
	}
}
