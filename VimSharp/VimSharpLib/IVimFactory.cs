namespace VimSharpLib;

public interface IVimFactory
{
    T CreateEditor<T>(IConsoleDevice consoleDevice)
        where T : VimEditor;

    T CreateVimMode<T>(VimEditor editor)
        where T : IVimMode;
}