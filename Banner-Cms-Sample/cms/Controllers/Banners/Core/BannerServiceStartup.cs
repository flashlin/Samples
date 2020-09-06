using System.Linq;
using Microsoft.Extensions.DependencyInjection;
using WebSample.Controllers.Banners.Providers;
using WebSample.Controllers.Banners.Repositories;
using T1.Standard.Extensions;
using WebSample.Controllers.Banners.Core;

namespace cms.Controllers.Banners.Core
{
    public static class BannerServiceStartup
    {
        public static void AddBannerService(this IServiceCollection services)
        {
            services.AddTransient<IBannerProviderFactory, BannerProviderFactory>();
            services.AddSingleton<IBannerRepo, BannerRepo>();

            AddAllBannerProviderTypesScoped(services);
        }

        private static void AddAllBannerProviderTypesScoped(IServiceCollection services)
        {
            var bannerProviderTypes = typeof(IBannerProvider).Assembly
                .GetTypes()
                .Where(x => typeof(IBannerProvider).IsAssignableFrom(x) && !x.IsInterface && !x.IsAbstract)
                .ToArray();

            foreach (var bannerProviderType in bannerProviderTypes)
            {
                services.AddScoped(bannerProviderType);
            }
        }
    }
}