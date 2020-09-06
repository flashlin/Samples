using System.Collections.Generic;
using WebSample.Controllers.Banners.Providers;

namespace WebSample.Controllers.Banners.Repositories
{
	public interface IBannerRepo
	{
		IEnumerable<IBanner> GetAllBanners(string bannerId);
	}
}