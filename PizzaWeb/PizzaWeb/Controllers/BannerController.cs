using Microsoft.AspNetCore.Mvc;
using PizzaWeb.Models;
using PizzaWeb.Models.Banner;
using T1.AspNetCore;
using T1.Standard.Common;

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

		public List<BannerTemplate> GetAllTemplates()
		{
			var bannerTemplates = _dbContext.BannerTemplates
				.ToList();
			return bannerTemplates.Select(x => ToBannerTemplateData(x)).ToList();
		}

		private BannerTemplate ToBannerTemplateData(BannerTemplateEntity entity)
		{
			var row = ValueHelper.CopyData(entity, new BannerTemplate());
			if (entity.VariablesData != null)
			{
				row.Variables = row.GetVariables(entity.VariablesData);
			}
			return row;
		}

		[HttpPost]
		public void UpdateTemplate(BannerTemplate req)
		{
			var bannerTemplate = _dbContext.BannerTemplates.Find(req.Id)!;
			bannerTemplate.LastModifiedTime = DateTime.Now;
			bannerTemplate.TemplateContent = req.TemplateContent;
			bannerTemplate.VariablesData = req.GetVariablesData();
			_dbContext.BannerTemplates.Update(bannerTemplate);
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

			var bannerData = allBannerData.Where(x => x.BannerName == req.BannerId)
				.ToList();

			var bannerLogical = new BannerLogical[]
			{

			};

			var content = await _viewToStringRenderer.RenderViewToStringAsync<object>(@$"/banner-template:/{req.BannerId}.banner-template",
				bannerData);
			return content;
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
		public string BannerId { get; set; } = "";
		public string LangCode { get; set; } = "";
	}
}
