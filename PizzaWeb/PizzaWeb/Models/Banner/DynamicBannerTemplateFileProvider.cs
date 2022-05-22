using System.Collections.Concurrent;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Options;
using T1.AspNetCore.FileProviders.Virtual;
using T1.Standard.Extensions;

namespace PizzaWeb.Models.Banner
{
	public class DynamicBannerTemplateFileProvider : IDynamicFileProvider
	{
		private readonly PizzaDbContext _db;
		ConcurrentDictionary<string, IFileInfo> _files = new ConcurrentDictionary<string, IFileInfo>();

		public DynamicBannerTemplateFileProvider(string connectionString)
		{
			var optionsFactory = new UseSqlServerByConnectionString(connectionString);
			_db = new PizzaDbContext(optionsFactory);
		}

		public IFileInfo GetFileInfo(string subPath)
		{
			var match = MatchPattern(subPath);
			if (!match.Success)
			{
				return new NotFoundFileInfo(subPath);
			}

			return _files.GetOrAdd(subPath, x =>
			{
				var templateId = match.Groups["id"].Value;
				var template = _db.BannerTemplates.FirstOrDefault(x => x.TemplateName == templateId);
				if (template == null)
				{
					return new NotFoundFileInfo(subPath);
				}
				var content = template.TemplateContent;
				return new VirtualFileInfo(subPath, template.TemplateName, DateTimeOffset.Now, false, (info) => Encoding.UTF8.GetBytes(content));
			});
		}

		public bool HasChanged(string subPath)
		{
			if (!_files.TryGetValue(subPath, out var fileInfo))
			{
				return true;
			}

			var match = MatchPattern(subPath);
			if (!match.Success)
			{
				return false;
			}
			var templateId = match.Groups["id"].Value;

			var lastModifiedTime = _db.BannerTemplates
				.Where(x => x.TemplateName == templateId)
				.Select(x => x.LastModifiedTime)
				.FirstOrDefault();

			if (lastModifiedTime == DateTime.MinValue)
			{
				return false;
			}

			return new DateTimeOffset(lastModifiedTime) > fileInfo.LastModified;
		}

		public bool? Match(string subPath)
		{
			if (!subPath.StartsWith("/banner-template:", StringComparison.OrdinalIgnoreCase))
			{
				return null;
			}

			return MatchPattern(subPath).Success;
		}

		private static Match MatchPattern(string subPath)
		{
			var pattern =
				"^/banner-template:/" +
				RegexPattern.Group("id", "[^/]+") +
				".banner-template$";
			var rg = new Regex(pattern);
			var match = rg.Match(subPath);
			return match;
		}

		private string GetFileName(string relativeUrl)
		{
			var baseUri = new Uri("http://www.mrbrain.com/");
			var myUri = new Uri(baseUri, relativeUrl);
			var fileName = Path.GetFileName(myUri.LocalPath);
			return fileName;
		}

		public void SignalChangeToken(string subPath)
		{
			var match = MatchPattern(subPath);
			if (!match.Success)
			{
				return;
			}
			var templateId = match.Groups["id"].Value;

			var template = _db.BannerTemplates.Where(x => x.TemplateName == templateId)
				.FirstOrDefault();
			if (template == null)
			{
				return;
			}

			template.LastModifiedTime = DateTime.Now;
			_db.SaveChanges();
		}
	}
}
