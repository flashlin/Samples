namespace VimSharpLib;

public class ConsoleRender
{
    public void Render(RenderArgs args)
    {
        Console.SetCursorPosition(args.X, args.Y);
        foreach (var c in args.Text.Chars)
        {
            Console.ForegroundColor = c.Color;
            Console.BackgroundColor = c.BackgroundColor;
            Console.Write(c.Value);
        }
        Console.ResetColor();
    }
}