using System.Collections.Generic;
using System.Linq;
using WebSample.Controllers.Banners.Providers;

namespace WebSample.Controllers.Banners.Repositories
{
	public class BannerRepo : IBannerRepo
	{
		public IEnumerable<IBanner> GetAllBanners(string bannerId)
		{
			return Enumerable.Empty<IBanner>();
		}
	}
}