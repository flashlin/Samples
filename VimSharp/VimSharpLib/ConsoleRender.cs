namespace VimSharpLib;

public class ConsoleRender
{
    // 預設前景色和背景色
    private ConsoleColor _defaultForeground = ConsoleColor.White;
    private ConsoleColor _defaultBackground = ConsoleColor.Black;

    public void Render(RenderArgs args)
    {
        Console.SetCursorPosition(args.X, args.Y);
        
        // 直接輸出帶顏色的文字
        Console.Write(args.Text.ToColoredString());
    }
    
    // 添加一個新方法，只渲染單個字符
    public void RenderChar(int x, int y, ColoredChar c)
    {
        Console.SetCursorPosition(x, y);
        
        // 直接輸出帶顏色的字元
        Console.Write(c.ToAnsiString());
    }
    
    // 添加一個新方法，渲染帶顏色的字元
    public void RenderColoredChar(int x, int y, char c, ConsoleColor foreground, ConsoleColor background)
    {
        Console.SetCursorPosition(x, y);
        
        // 創建 ColoredChar 並輸出
        var coloredChar = new ColoredChar(c, foreground, background);
        Console.Write(coloredChar.ToAnsiString());
    }
}