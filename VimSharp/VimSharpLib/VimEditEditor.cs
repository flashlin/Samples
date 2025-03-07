namespace VimSharpLib;

public class VimEditEditor
{
    ConsoleRender _render { get; set; } = new();
    public ConsoleContext Context { get; set; } = new();
    public void Run()
    {
        _render.Render(new RenderArgs
        {
            X = Context.X,
            Y = Context.Y,
            Text = Context.Texts[0]
        });
    }
    
    public void WaitForInput()
    {
        var keyInfo = Console.ReadKey(intercept: false); 
    }
}