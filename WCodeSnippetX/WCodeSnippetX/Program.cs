using System.Reflection;
using System.Runtime.CompilerServices;
using CefSharp;
using CefSharp.SchemeHandler;
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
			var host = CreateHostBuilder()
				.Build();

			ApplicationConfiguration.Initialize();
			InitializeCefSharp();

			var serviceProvider = host.Services;
			ConfigureApp(serviceProvider);
			host.RunAsync();
			//Application.Run(serviceProvider.GetRequiredService<FormMain>());
			Application.Run(serviceProvider.GetRequiredService<FormMainCef>());

			//ApplicationConfiguration.Initialize();
			//Application.Run(new FormMain());

			Cef.Shutdown();
		}

		public static IHostBuilder CreateHostBuilder() =>
			Host.CreateDefaultBuilder()
				.ConfigureWebHostDefaults(webBuilder =>
				{
					webBuilder.UseUrls("http://*:8880");
					webBuilder.UseStartup<Startup>();
				});

		static void ConfigureApp(IServiceProvider serviceProvider)
		{
			using var serviceScope = serviceProvider.GetService<IServiceScopeFactory>()!.CreateScope();
			var context = serviceScope.ServiceProvider.GetRequiredService<CodeSnippetDbContext>();
			context.Database.EnsureCreated();
		}

		[MethodImpl(MethodImplOptions.NoInlining)]
		private static void InitializeCefSharp()
		{
			var settings = new CefSettings();
			var baseDir = AppDomain.CurrentDomain.BaseDirectory;

			settings.RegisterScheme(new CefCustomScheme
			{
				SchemeName = "localfolder",
				DomainName = "cefsharp",
				SchemeHandlerFactory = new FolderSchemeHandlerFactory(
					rootFolder: $"{baseDir}/views",
					hostName: "cefsharp",
					defaultPage: "index.html"
				)
			});

			// Set BrowserSubProcessPath based on app bitness at runtime
			// .NET Core 註解這一行
			//settings.BrowserSubprocessPath = Path.Combine(AppDomain.CurrentDomain.SetupInformation.ApplicationBase,
			//	Environment.Is64BitProcess ? "x64" : "x86",
			//	"CefSharp.BrowserSubprocess.exe");

			Cef.Initialize(settings, false, browserProcessHandler: null);
		}

		// Will attempt to load missing assembly from either x86 or x64 subdir
		private static Assembly? Resolver(object sender, ResolveEventArgs args)
		{
			if (args.Name.StartsWith("CefSharp"))
			{
				var assemblyName = args.Name.Split(new[] { ',' }, 2)[0] + ".dll";
				var archSpecificPath = Path.Combine(AppDomain.CurrentDomain.SetupInformation.ApplicationBase!,
					Environment.Is64BitProcess ? "x64" : "x86",
					assemblyName);

				return File.Exists(archSpecificPath)
					? Assembly.LoadFile(archSpecificPath)
					: null;
			}

			return null;
		}
	}
}