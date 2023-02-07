using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Mvc.ApplicationParts;
using Microsoft.AspNetCore.Mvc.Razor.RuntimeCompilation;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Primitives;
using T1.WebTools.Controllers;

namespace T1.WebTools
{
	public static class Startup
	{
		/// <summary>
		/// 
		/// </summary>
		/// <param name="services"></param>
		/// <remarks><![CDATA[
		/// builder.Services.AddControllersWithViews();
		/// builder.Services.AddWebTools();
		/// ]]></remarks>
		public static void AddWebTools(this IServiceCollection services)
		{
			var assembly = typeof(ToolsController).Assembly;
			services.AddControllersWithViews()
				.AddApplicationPart(assembly)
				.AddRazorRuntimeCompilation();

			services.Configure<MvcRazorRuntimeCompilationOptions>(options =>
			{
				options.FileProviders.Add(new MyEmbeddedFileProvider(assembly)); 
				options.FileProviders.Add(new EmbeddedFileProvider(assembly)); 
			});
		}

        /// <summary>
        /// 
        /// </summary>
        /// <param name="app"></param>
        /// <remarks><![CDATA[
        /// app.UseWebTools(); //must
        /// app.UseStaticFiles();
		/// ]]></remarks>
        public static void UseWebTools(this IApplicationBuilder app)
        {
            var assembly = typeof(ToolsController).Assembly;
            app.UseStaticFiles(new StaticFileOptions
            {
                FileProvider = new MyEmbeddedFileProvider(assembly)
            });
        }
	}

	public class MyEmbeddedFileProvider : IFileProvider
	{
		private readonly Assembly _assembly;
		public MyEmbeddedFileProvider(Assembly assembly)
		{
			_assembly = assembly;
		}

		public IFileInfo GetFileInfo(string subpath)
		{
			var prefix = "/component:/";
			if (!subpath.StartsWith(prefix))
			{
				return new NotFoundFileInfo(subpath);
			}

			var embeddedName = subpath.Substring(prefix.Length);
			embeddedName = embeddedName.Replace("/", ".");
			var embeddedFullName =_assembly.GetManifestResourceNames()
                .FirstOrDefault(x => x.EndsWith(embeddedName));
            if (embeddedFullName == null)
            {
				return new NotFoundFileInfo(subpath);
            }

			return new EmbeddedFileInfo(_assembly, embeddedFullName);
		}

		public IDirectoryContents GetDirectoryContents(string subpath)
		{
			return new NotFoundDirectoryContents();
		}

		public IChangeToken Watch(string filter)
		{
			return NullChangeToken.Singleton;
		}
	}


	public class EmbeddedFileInfo : IFileInfo
	{
		private readonly string _embeddedFullName;
        private readonly Stream _stream;

        public EmbeddedFileInfo(Assembly assembly, string embeddedFullName)
        {
            _embeddedFullName = embeddedFullName;
            _stream = assembly.GetManifestResourceStream(_embeddedFullName)!;
        }

		public bool Exists => true;

		public long Length => _stream.Length;

		public string PhysicalPath => string.Empty;

		public string Name => Path.GetFileNameWithoutExtension(_embeddedFullName);

		public DateTimeOffset LastModified => DateTimeOffset.Now;//DateTimeOffset.MinValue;

		public bool IsDirectory => false;

		public Stream CreateReadStream()
		{
            return _stream;
        }
	}

}
