﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using T1.Standard.Serialization;
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
			services.AddTransient<ICodeSnippetService, CodeSnippetService>();
			services.AddTransient<IBoundObject, BoundObject>();
			services.AddTransient<IJsonSerializer, MyJsonSerializer>();
		}

		public void Configure(IApplicationBuilder app)
		{
			app.UseRouting();
			app.UseEndpoints(endpoints =>
			{
				endpoints.MapControllers();
			});
		}

		public void ConfigureApp(IServiceProvider serviceProvider)
		{
			using var serviceScope = serviceProvider.GetService<IServiceScopeFactory>()!.CreateScope();
			var context = serviceScope.ServiceProvider.GetRequiredService<CodeSnippetDbContext>();
			context.Database.EnsureCreated();
		}
	}
}
