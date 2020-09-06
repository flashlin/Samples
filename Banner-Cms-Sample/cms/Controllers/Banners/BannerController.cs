using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using T1.AspNetCore;
using T1.Standard.Common;
using T1.Standard.Extensions;
using WebSample.Controllers.Banners.Core;

namespace WebSample.Controllers.Banners
{
	[Route("api/[controller]/[action]")]
	[ApiController]
	public class BannerController : ControllerBase
	{
		private readonly IViewToStringRendererService _viewRenderer;
		private readonly IBannerProviderFactory _bannerProviderFactory;

		public BannerController(IViewToStringRendererService viewRenderer,
			IBannerProviderFactory bannerProviderFactory)
		{
			_bannerProviderFactory = bannerProviderFactory;
			_viewRenderer = viewRenderer;
		}

		//http://localhost:5000/api/Banner/Get?BannerId=Product
		public Task<string> Get(BannerRequest req)
		{
			var temp = $"/Controllers/Banners/Templates/{req.BannerId}.cshtml";
			var viewModel = _bannerProviderFactory.Create(req.BannerId)
				.Build(req);
			return _viewRenderer.RenderViewToStringAsync(temp, viewModel);
		}
	}

}
