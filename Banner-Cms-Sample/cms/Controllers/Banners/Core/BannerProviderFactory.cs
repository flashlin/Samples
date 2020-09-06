using System;
using WebSample.Controllers.Banners.Providers;

namespace WebSample.Controllers.Banners.Core
{
	public class BannerProviderFactory : IBannerProviderFactory
	{
		private readonly IServiceProvider _serviceProvider;

		public BannerProviderFactory(IServiceProvider serviceProvider)
		{
			_serviceProvider = serviceProvider;
		}

		public IBannerProvider Create(string bannerId)
		{
			var providerNamespace = typeof(IBannerProvider).Namespace;
			var bannerProviderType = Type.GetType($"{providerNamespace}.{bannerId}Provider");
			return (IBannerProvider)_serviceProvider.GetService(bannerProviderType);
		}
	}
}