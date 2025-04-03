using Microsoft.Extensions.DependencyInjection;
using NSubstitute;

namespace VimSharpLib.Tests;

public class VimSharpTester
{
    private readonly ServiceCollection _services = new();
    private ServiceProvider _serviceProvider;

    public VimSharpTester()
    {
        MockConsole = Substitute.For<IConsoleDevice>();
        MockConsole.WindowWidth.Returns(80);
        MockConsole.WindowHeight.Returns(25);
        ScreenBuffer = ColoredCharScreen.CreateScreenBuffer(MockConsole);
        Build();
    }
    
    public IConsoleDevice MockConsole { get; }
    public ColoredCharScreen ScreenBuffer { get; }

    public ServiceProvider Build()
    {
        _services.AddVimSharpServices();
        _services.AddTransient<IConsoleDevice>(_ => MockConsole);
        _serviceProvider = _services.BuildServiceProvider();
        return _serviceProvider;
    }

    public T GetRequiredService<T>()
        where T : class 
    {
        return _serviceProvider.GetRequiredService<T>();
    }
    
    public VimEditor CreateVimEditor()
    {
        var vimEditor = _serviceProvider.GetRequiredService<VimEditor>();
        vimEditor.Console = MockConsole;
        return vimEditor;
    }
}