using System.Collections.Generic;
using System.Linq;
using WebSample.Controllers.Banners.Core;
using WebSample.Controllers.Banners.Repositories;
using WebSample.Controllers.Banners.ViewModels;

namespace WebSample.Controllers.Banners.Providers
{
	public class HeroBannerProvider : IBannerProvider
	{
		private IBannerRepo _bannerRepo;

		public HeroBannerProvider(IBannerRepo bannerRepo)
		{
			_bannerRepo = bannerRepo;
		}



		public IViewModel Build(BannerRequest req)
		{
			var banners = _bannerRepo.GetAllBanners("HeroBannerId");
			banners = FilterBanners(banners, req);
			return new HeroBannerViewModel()
			{
				Banners = banners.Cast<HeroBannerViewModel>()
			};
		}

		private IEnumerable<IBanner> FilterBanners(IEnumerable<IBanner> banners, BannerRequest req)
		{
			return banners;
		}
	}
}