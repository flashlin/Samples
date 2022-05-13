using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using PizzaWeb.Models;

namespace PizzaWeb.Controllers
{
	[Route("api/[controller]/[action]")]
	[ApiController]
	public class StoreShelvesController : ControllerBase
	{
		private readonly PizzaDbContext _dbContext;
		private readonly IWebHostEnvironment _hostEnvironment;

		public StoreShelvesController(PizzaDbContext dbContext, IWebHostEnvironment hostEnvironment)
		{
			this._dbContext = dbContext;
			this._hostEnvironment = hostEnvironment;
		}

		[HttpGet]
		public List<StoreShelvesEntity> GetAll()
		{
			return _dbContext.StoreShelves.ToList();
		}

		[HttpPost]
		public void Modify(StoreShelvesEntity entity)
		{
			_dbContext.StoreShelves.Update(entity);
			_dbContext.SaveChanges();
		}

		[HttpPost]
		public void SaveBlob([FromForm]BlobDto req)
		{
			var file = req.Images.First();

			var path = Path.Combine(_hostEnvironment.WebRootPath, "images");
			var filePath = Path.Combine(path, $"img{req.ImageId}.jpg");
			using var stream = req.IsFirst ? new FileStream(filePath, FileMode.Create) : new FileStream(filePath, FileMode.Open);
			stream.Seek(0, SeekOrigin.End);

			using var ms = new MemoryStream();
			file.CopyTo(ms);
			var fileBytes = ms.ToArray();
			stream.Write(fileBytes);
			stream.Flush();
		}
	}

	public class BlobDto
	{
		public int ImageId { get; set; }
		[FromForm(Name = "image")]
		public List<IFormFile> Images { get; set; }
		public bool IsFirst { get; set; }
	}
}
