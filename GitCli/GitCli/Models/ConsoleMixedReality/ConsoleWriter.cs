namespace GitCli.Models.ConsoleMixedReality;

public class ConsoleWriter : IConsoleWriter
{
    private readonly ConsoleColor _originBackgroundColor;
    private readonly ConsoleColor _originForegroundColor;
    private ConsoleColor _foregroundColor;
    private ConsoleColor _backgroundColor;

    public ConsoleWriter()
    {
        _originBackgroundColor = Console.BackgroundColor;
        _backgroundColor = _originBackgroundColor;
        _originForegroundColor = Console.ForegroundColor;
        _foregroundColor = _originForegroundColor;
    }

    public ConsoleSize GetSize()
    {
        return new ConsoleSize()
        {
            Width = Console.WindowWidth,
            Height = Console.WindowHeight,
        };
    }

    public void SetForegroundColor(ConsoleColor color)
    {
        _foregroundColor = color;
        Console.ForegroundColor = color;
    }
    
    public void SetBackgroundColor(ConsoleColor color)
    {
        _backgroundColor = color;
        Console.BackgroundColor = color;
    }

    public void Write(string text)
    {
        Console.Write(text);
    }

    public void WriteLine(string text)
    {
        Console.WriteLine(text);
    }

    public void ResetWriteColor()
    {
        Console.BackgroundColor = _backgroundColor;
        Console.ForegroundColor = _foregroundColor;
    }

    public void ResetColor()
    {
        Console.BackgroundColor = _originBackgroundColor;
        Console.ForegroundColor = _originForegroundColor;
    }
}