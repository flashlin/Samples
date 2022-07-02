namespace GitCli.Models.ConsoleMixedReality;

public static class ConsoleElementExtension
{
    public static T Setup<T>(this T elem, Action<T> setupFn)
        where T : IConsoleElement
    {
        setupFn(elem);
        return elem;
    }
}