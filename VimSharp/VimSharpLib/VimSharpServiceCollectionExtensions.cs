using Microsoft.Extensions.DependencyInjection;

namespace VimSharpLib;

public static class VimSharpServiceCollectionExtensions
{
    public static IServiceCollection AddVimSharpServices(this IServiceCollection services)
    {
        services.AddSingleton<IConsoleDevice, ConsoleDevice>();
        services.AddTransient<IKeyHandler, KeyHandler>();
        services.AddTransient<IVimFactory, VimFactory>();
        services.AddTransient<VimNormalMode>();
        services.AddTransient<VimInsertMode>();
        services.AddTransient<VimVisualMode>();
        services.AddTransient<VimCommandMode>();
        services.AddTransient<VimEditor>();
        services.AddTransient<VimCommand>();
        return services;
    }
}