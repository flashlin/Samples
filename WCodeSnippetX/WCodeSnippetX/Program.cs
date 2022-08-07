using CefSharp;
using CefSharp.WinForms;
using Microsoft.Extensions.DependencyInjection;
using WCodeSnippetX.Models;
using WCodeSnippetX.Models.Repos;
using WCodeSnippetX.ViewComponents;

namespace WCodeSnippetX
{
	internal static class Program
	{
		/// <summary>
		///  The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main()
		{
			//Application.SetHighDpiMode(HighDpiMode.SystemAware);
			//Application.EnableVisualStyles();
			//Application.SetCompatibleTextRenderingDefault(false);

			var services = new ServiceCollection();
			ConfigureServices(services);
			ApplicationConfiguration.Initialize();

			var settings = new CefSettings();
			Cef.Initialize(settings, false, browserProcessHandler: null);

			using var serviceProvider = services.BuildServiceProvider();
			ConfigureApp(serviceProvider);
			//Application.Run(serviceProvider.GetRequiredService<FormMain>());
			Application.Run(serviceProvider.GetRequiredService<FormMainCef>());

			//ApplicationConfiguration.Initialize();
			//Application.Run(new FormMain());

			Cef.Shutdown();
		}

		static void ConfigureServices(ServiceCollection services)
		{
			services.AddScoped<FormMain>();
			services.AddScoped<FormMainCef>();
			services.AddScoped<FormEditCode>();
			services.AddDbContext<CodeSnippetDbContext>();
			services.AddTransient<ICodeSnippetRepo, CodeSnippetRepo>();
		}

		static void ConfigureApp(IServiceProvider serviceProvider)
		{
			using var serviceScope = serviceProvider.GetService<IServiceScopeFactory>()!.CreateScope();
			var context = serviceScope.ServiceProvider.GetRequiredService<CodeSnippetDbContext>();
			context.Database.EnsureCreated();
		}
	}
}