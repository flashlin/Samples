using CommunityToolkit.Maui.Markup;
using GitMaui.ViewModels;

namespace GitMaui;

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
				fonts.AddFont("Font Awesome 6 Free-Regular-400.otf", "FontAwesome");
			});

		builder.Services.AddSingleton<MainViewModel>();
		
		builder.Services.AddSingleton<MainPage>();

		return builder.Build();
	}
}
