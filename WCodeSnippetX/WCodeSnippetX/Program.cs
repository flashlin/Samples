using System.Reflection;
using System.Runtime.CompilerServices;
using CefSharp;
using CefSharp.SchemeHandler;
using CefSharp.WinForms;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
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
			//create dotnet self-host
			//var host = CreateHostBuilder()
			//	.Build();
			//var serviceProvider = host.Services;

			var configurationBuilder = new ConfigurationBuilder();
			configurationBuilder.SetBasePath(Directory.GetCurrentDirectory());
			var configuration = configurationBuilder.Build();
			var startup = new Startup(configuration);
			var services = new ServiceCollection();
			startup.ConfigureServices(services);

			//host.RunAsync();
			ApplicationConfiguration.Initialize();
			InitializeCefSharp();

			using var serviceProvider = Build(services);
			startup.ConfigureApp(serviceProvider);

			//var server = host.Services.GetService<IServer>()!;
			//var addressFeature = server.Features.Get<IServerAddressesFeature>();
			//var address = addressFeature.Addresses.First();
			//var idx = address.LastIndexOf(":");
			//var port = address.Substring(idx+1);

			//Application.Run(serviceProvider.GetRequiredService<FormMain>());
			Application.Run(serviceProvider.GetRequiredService<FormMainCef>());

			//ApplicationConfiguration.Initialize();
			//Application.Run(new FormMain());

			Cef.Shutdown();
		}

		private static ServiceProvider Build(ServiceCollection services)
		{
			return services.BuildServiceProvider();
		}

		public static IHostBuilder CreateHostBuilder() =>
			Host.CreateDefaultBuilder()
				.ConfigureWebHostDefaults(webBuilder =>
				{
					webBuilder.UseUrls("http://*:8880");
					webBuilder.UseStartup<Startup>();
				});

		[MethodImpl(MethodImplOptions.NoInlining)]
		private static void InitializeCefSharp()
		{
			var settings = new CefSettings()
			{
				RemoteDebuggingPort = 8888
			};
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
			// .NET Core ���ѳo�@��
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