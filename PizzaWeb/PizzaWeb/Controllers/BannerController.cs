using Microsoft.AspNetCore.Mvc;
using PizzaWeb.Models;
using PizzaWeb.Models.Banner;
using T1.AspNetCore;

namespace PizzaWeb.Controllers
{
	[Route("api/[controller]/[action]")]
	[ApiController]
	public class BannerController : ControllerBase
	{
		private PizzaDbContext _dbContext;
		private IViewToStringRendererService _viewToStringRenderer;

		public BannerController(PizzaDbContext dbContext, IViewToStringRendererService viewToStringRenderer)
		{
			_viewToStringRenderer = viewToStringRenderer;
			_dbContext = dbContext;
		}

		public List<BannerTemplateEntity> GetAllTemplates()
		{
			return _dbContext.BannerTemplates.ToList();
		}

		[HttpPost]
		public void UpdateTemplate(BannerTemplateEntity req)
		{
			_dbContext.BannerTemplates.Update(req);
			_dbContext.SaveChanges();
		}

		public async Task<string> GetBanner(GetBannerReq req)
		{
			var allBannerData = new[] {
				new BannerData()
				{
					LangCode = "TW",
					BannerName = "A1",
				},
				new BannerData()
				{
					LangCode = "CN",
					BannerName = "A1",
				},
				new BannerData()
				{
					LangCode = "EN",
					BannerName = "A2",
				},
			};

			var bannerData = allBannerData.Where(x => x.BannerName == req.BannerName)
				.ToList();

			var bannerLogical = new BannerLogical[]
			{

			};


			var html = Render(bannerData);
			var text = await _viewToStringRenderer.RenderViewToStringAsync<object>("/Files/Hello.cshtml", bannerData);
			return String.Empty;
		}

		private string Render(List<BannerData> bannerData)
		{
			return string.Empty;
		}
	}

	public class BannerData
	{
		public string LangCode { get; set; } = "";
		public string BannerName { get; set; } = "";
	}

	public class BannerLogical
	{
		public string BannerName { get; set; } = "";
		public string Code { get; set; } = "";
	}

	public class GetBannerReq
	{
		public string BannerName { get; set; } = "";
		public string LangCode { get; set; } = "";
	}
}
