using Microsoft.Extensions.DependencyInjection;
using WCodeSnippetX.Models.Repos;

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
			using var serviceProvider = services.BuildServiceProvider();
			var form = serviceProvider.GetRequiredService<FormMain>();
			Application.Run(form);

			//ApplicationConfiguration.Initialize();
			//Application.Run(new FormMain());
		}

		static void ConfigureServices(ServiceCollection services)
		{
			services.AddScoped<FormMain>();
			services.AddDbContext<CodeSnippetDbContext>();
		}
	}
}