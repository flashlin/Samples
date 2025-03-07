namespace VimSharpLib;

public class VimEditor
{
    ConsoleRender _render { get; set; } = new();
    public ConsoleContext Context { get; set; } = new();

    public void Initialize()
    {
    }

    public void Run()
    {
        Render();
        WaitForInput();
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
    
    public void WaitForInput()
    {
        while (true)
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
                
                // 編輯模式結束後重新渲染
                Render();
            }
            else if (keyInfo.Key == ConsoleKey.Q)
            {
                // 退出編輯器
                break;
            }
        }
    }
}