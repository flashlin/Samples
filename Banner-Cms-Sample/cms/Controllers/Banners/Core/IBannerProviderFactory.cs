using WebSample.Controllers.Banners.Providers;

namespace WebSample.Controllers.Banners.Core
{
	public interface IBannerProviderFactory
	{
		IBannerProvider Create(string bannerId);
	}
}