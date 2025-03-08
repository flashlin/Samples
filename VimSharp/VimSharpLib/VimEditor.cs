namespace VimSharpLib;

public class VimEditor
{
    ConsoleRender _render { get; set; } = new();

    public VimEditor()
    {
        Initialize();
    }

    public bool IsRunning { get; set; } = true;
    public ConsoleContext Context { get; set; } = new();
    public IVimMode Mode { get; set; }

    public void Initialize()
    {
        Context.SetText(0, 0, "Hello, World!");
        
        // 設置 ViewPort 的初始值為部分控制台視窗
        Context.ViewPort = new ConsoleRectangle(10, 10, Console.WindowWidth - 20, Console.WindowHeight - 20);
        
        Mode = new VimVisualMode { Instance = this };
    }

    public void Run()
    {
        while (IsRunning)
        {
            Render();
            WaitForInput();
        }
    }

    public void Render()
    {
        // 獲取控制台當前寬度和高度
        var width = Console.WindowWidth;
        var height = Console.WindowHeight;
        for(var y=0; y<height; y++)
        {
            var text = Context.GetText(y);
            text.Width = width;
            _render.Render(new RenderArgs
            {
                X = Context.ViewPort.X,
                Y = Context.ViewPort.Y + y,
                Text = text,
                ViewPort = Context.ViewPort,
            });
        }

        Console.SetCursorPosition(Context.CursorX, Context.CursorY);
    }

    public void WaitForInput()
    {
        Mode.WaitForInput();
    }
}