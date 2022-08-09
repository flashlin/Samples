using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using WCodeSnippetX.Models;
using WCodeSnippetX.Models.Repos;
using WCodeSnippetX.ViewComponents;

namespace WCodeSnippetX
{
	public class Startup
	{
		public Startup(IConfiguration configuration)
		{
			Configuration = configuration;
		}

		public IConfiguration Configuration { get; }

		public void ConfigureServices(IServiceCollection services)
		{
			services.AddControllersWithViews();
			//services.AddControllers();
			services.AddScoped<FormMain>();
			services.AddSingleton<FormMainCef>();
			services.AddScoped<FormEditCode>();
			services.AddDbContext<CodeSnippetDbContext>();
			services.AddTransient<ICodeSnippetRepo, CodeSnippetRepo>();
			services.AddSingleton<IBoundObject, BoundObject>();
		}

		public void Configure(IApplicationBuilder app)
		{
			app.UseRouting();


			app.UseEndpoints(endpoints =>
			{
				endpoints.MapControllers();
			});
		}
	}
}
