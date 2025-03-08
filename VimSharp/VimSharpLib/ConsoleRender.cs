namespace VimSharpLib;
using System.Text;

public class ConsoleRender
{
    // 預設前景色和背景色
    private ConsoleColor _defaultForeground = ConsoleColor.White;
    private ConsoleColor _defaultBackground = ConsoleColor.Black;

    public void Render(RenderArgs args)
    {
        // 檢查 Y 座標是否在 ViewPort 範圍內
        if (args.Y < args.ViewPort.Y || args.Y >= args.ViewPort.Y + args.ViewPort.Height)
        {
            return; // Y 座標超出範圍，不繪製
        }

        // 計算可見的起始和結束位置
        int startX = Math.Max(0, args.ViewPort.X);
        int endX = Math.Min(args.Text.Chars.Length, args.ViewPort.X + args.ViewPort.Width);
        
        // 如果起始位置已經超出文本範圍或結束位置小於等於起始位置，則不繪製
        if (startX >= args.Text.Chars.Length || endX <= startX)
        {
            return;
        }
        
        // 設置光標位置到可見區域的起始位置
        Console.SetCursorPosition(Math.Max(args.X, args.ViewPort.X), args.Y);
        
        // 只繪製可見範圍內的文本
        var sb = new StringBuilder();
        for (int i = startX; i < endX; i++)
        {
            var c = args.Text.Chars[i];
            if (c.Char == '\0')
            {
                continue;
            }
            sb.Append(c.ToAnsiString());
        }
        
        Console.Write(sb.ToString());
    }
    
    // 添加一個新方法，只渲染單個字符
    public void RenderChar(int x, int y, ColoredChar c, ConsoleRectangle viewPort)
    {
        // 檢查座標是否在 ViewPort 範圍內
        if (x < viewPort.X || x >= viewPort.X + viewPort.Width ||
            y < viewPort.Y || y >= viewPort.Y + viewPort.Height)
        {
            return; // 座標超出範圍，不繪製
        }
        
        Console.SetCursorPosition(x, y);
        
        // 直接輸出帶顏色的字元
        Console.Write(c.ToAnsiString());
    }
    
    // 添加一個新方法，渲染帶顏色的字元
    public void RenderColoredChar(int x, int y, char c, ConsoleColor foreground, ConsoleColor background, ConsoleRectangle viewPort)
    {
        // 檢查座標是否在 ViewPort 範圍內
        if (x < viewPort.X || x >= viewPort.X + viewPort.Width ||
            y < viewPort.Y || y >= viewPort.Y + viewPort.Height)
        {
            return; // 座標超出範圍，不繪製
        }
        
        Console.SetCursorPosition(x, y);
        
        // 創建 ColoredChar 並輸出
        var coloredChar = new ColoredChar(c, foreground, background);
        Console.Write(coloredChar.ToAnsiString());
    }
    
    // 保留原來的方法以保持向後兼容性
    public void RenderChar(int x, int y, ColoredChar c)
    {
        // 創建一個默認的 ViewPort，包含整個控制台
        var viewPort = new ConsoleRectangle(0, 0, Console.WindowWidth, Console.WindowHeight);
        RenderChar(x, y, c, viewPort);
    }
    
    // 保留原來的方法以保持向後兼容性
    public void RenderColoredChar(int x, int y, char c, ConsoleColor foreground, ConsoleColor background)
    {
        // 創建一個默認的 ViewPort，包含整個控制台
        var viewPort = new ConsoleRectangle(0, 0, Console.WindowWidth, Console.WindowHeight);
        RenderColoredChar(x, y, c, foreground, background, viewPort);
    }
}