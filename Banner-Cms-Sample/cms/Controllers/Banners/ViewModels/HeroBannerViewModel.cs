using System.Collections.Generic;
using WebSample.Controllers.Banners.Core;
using WebSample.Controllers.Banners.Providers;

namespace WebSample.Controllers.Banners.ViewModels
{
	public class HeroBannerViewModel : IViewModel
	{
		public IEnumerable<HeroBannerViewModel> Banners { get; set; }
	}
}