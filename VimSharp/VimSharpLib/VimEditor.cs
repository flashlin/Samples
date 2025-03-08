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
            var textWidth = text.Width;
            text.Width = width;
            _render.Render(new RenderArgs
            {
                X = 0,
                Y = y,
                Text = text
            });
        }

        Console.SetCursorPosition(Context.CursorX, Context.CursorY);
    }

    public void WaitForInput()
    {
        Mode.WaitForInput();
    }
}