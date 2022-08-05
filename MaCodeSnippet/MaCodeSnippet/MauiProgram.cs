using CommunityToolkit.Maui.Markup;
using MaCodeSnippet.Models;
using MaCodeSnippet.ViewModels;

namespace MaCodeSnippet;

public static class MauiProgram
{
	public static MauiApp CreateMauiApp()
	{
		var builder = MauiApp.CreateBuilder();
		builder
			.UseMauiApp<App>()
			.UseMauiCommunityToolkitMarkup()
			.ConfigureFonts(fonts =>
			{
				fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
				fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
			});

		builder.Services.AddTransient<ICodeSnippetService, CodeSnippetService>();
		builder.Services.AddSingleton<CodeSnippetViewModel>();

		return builder.Build();
	}
}
