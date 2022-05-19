using Microsoft.AspNetCore.Mvc;
using PizzaWeb.Models;
using PizzaWeb.Models.Banner;

namespace PizzaWeb.Controllers
{
	[Route("api/[controller]/[action]")]
	[ApiController]
	public class BannerController : ControllerBase
	{
		private PizzaDbContext _dbContext;

		public BannerController(PizzaDbContext dbContext)
		{
			_dbContext = dbContext;
		}

		public List<BannerTemplateEntity> GetAllTemplates()
		{
			return _dbContext.BannerTemplates.ToList();
		}

		public string GetBanner(GetBannerReq req)
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
			return String.Empty;
		}

		private string Render(List<BannerData> bannerData)
		{
			return string.Empty;
		}
	}

	public class BannerData
	{
		public string LangCode { get; set; }
		public string BannerName { get; set; }
	}

	public class BannerLogical
	{
		public string BannerName { get; set; }
		public string Code { get; set; }
	}

	public class GetBannerReq
	{
		public string BannerName { get; set; }
		public string LangCode { get; set; }
	}
}
