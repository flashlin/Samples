using System.Collections.Generic;
using System.Linq;
using WebSample.Controllers.Banners.Core;
using WebSample.Controllers.Banners.Repositories;
using WebSample.Controllers.Banners.ViewModels;

namespace WebSample.Controllers.Banners.Providers
{
	public class ProductBannerProvider : IBannerProvider
	{
		private readonly IBannerRepo _bannerRepo;

		public ProductBannerProvider(IBannerRepo bannerRepo)
		{
			_bannerRepo = bannerRepo;
		}

		public IEnumerable<IBanner> FilterTopBanner(IEnumerable<IBanner> banners, BannerRequest req)
		{
			return banners;
		}
		
		public IEnumerable<IBanner> FilterBottomBanner(IEnumerable<IBanner> banners, BannerRequest req)
		{
			return banners;
		}

		public IViewModel Build(BannerRequest req)
		{
			var topBanners = _bannerRepo.GetAllBanners($"Product-TopBanner");
			topBanners = FilterTopBanner(topBanners, req);

			var bottomBanners = _bannerRepo.GetAllBanners("Product-BottomBanner");
			bottomBanners = FilterBottomBanner(bottomBanners, req);

			return new ProductBannerViewModel()
			{
				TopBanners = topBanners.Cast<ProductTopBanner>(),
				BottomBanners = bottomBanners.Cast<ProductBottomBanner>()
			};
		}
	}
}
