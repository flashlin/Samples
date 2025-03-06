namespace VimSharpLib;

public class VimEditor
{
    ConsoleRender _render { get; set; } = new();
    public ConsoleContext Context { get; set; } = new();

    public void Initialize()
    {
    }

    public void Render()
    {
        Context.SetText(0, 0, "Hello, World!");
        _render.Render(new RenderArgs
        {
            X = 0,
            Y = 0,
            Text = Context.Texts[0]
        });
    }
}