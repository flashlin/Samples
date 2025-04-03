using Microsoft.Extensions.DependencyInjection;

namespace VimSharpLib;

public class VimFactory : IVimFactory
{
    private readonly IServiceProvider _serviceProvider;

    public VimFactory(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public T CreateEditor<T>(IConsoleDevice consoleDevice)
        where T : VimEditor
    {
        var vimEditor = _serviceProvider.GetRequiredService<T>();
        vimEditor.Console = consoleDevice;
        return vimEditor;
    }
    
    public T CreateVimMode<T>(VimEditor editor)
        where T : IVimMode
    {
        var vimMode = _serviceProvider.GetRequiredService<T>();
        vimMode.Instance = editor;
        return vimMode;
    }
}