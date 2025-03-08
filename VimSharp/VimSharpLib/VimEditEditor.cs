namespace VimSharpLib;
using System.Text;

public class VimEditEditor
{
    ConsoleRender _render { get; set; } = new();
    bool _continueEditing = true;
    public ConsoleContext Context { get; set; } = new();
    
    public void Run()
    {
        // 初始渲染
        // Console.Clear();
        
        
        if (Context.Texts.Count > 0)
        {
            _render.Render(new RenderArgs
            {
                X = 0,
                Y = 0,
                Text = Context.Texts[0]
            });
            Console.SetCursorPosition(Context.X, Context.Y);
        }
        
        // 創建並使用 VimNormalMode
        var normalMode = new VimNormalMode
        {
            Instance = new VimEditor { Context = this.Context }
        };
        
        while (_continueEditing)
        {
            normalMode.WaitForInput();
        }
    }
}