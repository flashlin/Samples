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
    
    // 添加一個新方法，只渲染單個字符
    public void RenderChar(int x, int y, ConsoleCharacter c)
    {
        Console.SetCursorPosition(x, y);
        Console.ForegroundColor = c.Color;
        Console.BackgroundColor = c.BackgroundColor;
        Console.Write(c.Value);
        Console.ResetColor();
    }
}