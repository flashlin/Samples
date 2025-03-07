namespace VimSharpLib;

public class VimEditor
{
    ConsoleRender _render { get; set; } = new();
    public ConsoleContext Context { get; set; } = new();

    public void Initialize()
    {
        Context.SetText(0, 0, "Hello, World!");
    }

    public void Run()
    {
        while (true)
        {
            Render();
            WaitForInput();
        }
    }

    public void Render()
    {
        _render.Render(new RenderArgs
        {
            X = 0,
            Y = 0,
            Text = Context.Texts[0]
        });
        Console.SetCursorPosition(Context.X, Context.Y);
    }

    public void WaitForInput()
    {
        var keyInfo = Console.ReadKey(intercept: true);

        if (keyInfo.Key == ConsoleKey.I)
        {
            // 進入編輯模式
            var editor = new VimEditEditor
            {
                Context = Context
            };
            editor.Run();
        }
    }
}