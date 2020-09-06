using System.Collections.Generic;
using WebSample.Controllers.Banners.Core;
using WebSample.Controllers.Banners.Providers;

namespace WebSample.Controllers.Banners.ViewModels
{
	public class ProductBannerViewModel : IViewModel
	{
		public IEnumerable<ProductTopBanner> TopBanners { get; set; }
		public IEnumerable<ProductBottomBanner> BottomBanners { get; set; }
	}
}