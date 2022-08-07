using System.Reflection;
using System.Runtime.CompilerServices;
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

			InitializeCefSharp();

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

		[MethodImpl(MethodImplOptions.NoInlining)]
		private static void InitializeCefSharp()
		{
			var settings = new CefSettings();

			// Set BrowserSubProcessPath based on app bitness at runtime
			// .NET Core 註解這一行
			//settings.BrowserSubprocessPath = Path.Combine(AppDomain.CurrentDomain.SetupInformation.ApplicationBase,
			//	Environment.Is64BitProcess ? "x64" : "x86",
			//	"CefSharp.BrowserSubprocess.exe");

			// Make sure you set performDependencyCheck false
			Cef.Initialize(settings, false, browserProcessHandler: null);
		}

		// Will attempt to load missing assembly from either x86 or x64 subdir
		private static Assembly Resolver(object sender, ResolveEventArgs args)
		{
			if (args.Name.StartsWith("CefSharp"))
			{
				var assemblyName = args.Name.Split(new[] { ',' }, 2)[0] + ".dll";
				var archSpecificPath = Path.Combine(AppDomain.CurrentDomain.SetupInformation.ApplicationBase,
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