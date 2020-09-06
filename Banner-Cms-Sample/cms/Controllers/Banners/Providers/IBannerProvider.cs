using System.Collections.Generic;
using WebSample.Controllers.Banners.Core;

namespace WebSample.Controllers.Banners.Providers
{
	public interface IBannerProvider
	{
		IViewModel Build(BannerRequest req);
	}
}