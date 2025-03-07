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
        
        WaitForInput();
    }
    
    public void WaitForInput()
    {
        bool continueEditing = true;
        
        while (continueEditing)
        {
            var keyInfo = Console.ReadKey(intercept: true);
            
            // 確保當前行存在
            if (Context.Texts.Count <= Context.Y)
            {
                Context.Texts.Add(new ConsoleText());
            }
            
            switch (keyInfo.Key)
            {
                case ConsoleKey.Escape:
                    continueEditing = false;
                    break;
                    
                case ConsoleKey.Backspace:
                    if (Context.X > 0)
                    {
                        // 獲取當前行
                        var currentLine = Context.Texts[Context.Y];
                        
                        // 刪除字符
                        string currentText = new string(currentLine.Chars.Select(c => c.Value).ToArray());
                        string newText = currentText.Remove(Context.X - 1, 1);
                        
                        // 更新文本
                        currentLine.SetText(0, newText);
                        
                        // 移動光標
                        Context.X--;
                    }
                    break;
                    
                case ConsoleKey.LeftArrow:
                    if (Context.X > 0)
                    {
                        Context.X--;
                    }
                    break;
                    
                case ConsoleKey.RightArrow:
                    var currentLineForRight = Context.Texts[Context.Y];
                    if (Context.X < currentLineForRight.Width)
                    {
                        Context.X++;
                    }
                    break;
                    
                case ConsoleKey.UpArrow:
                    if (Context.Y > 0)
                    {
                        Context.Y--;
                        // 確保 X 不超過新行的長度
                        if (Context.X > Context.Texts[Context.Y].Width)
                        {
                            Context.X = Context.Texts[Context.Y].Width;
                        }
                    }
                    break;
                    
                case ConsoleKey.DownArrow:
                    if (Context.Y < Context.Texts.Count - 1)
                    {
                        Context.Y++;
                        // 確保 X 不超過新行的長度
                        if (Context.X > Context.Texts[Context.Y].Width)
                        {
                            Context.X = Context.Texts[Context.Y].Width;
                        }
                    }
                    break;
                    
                default:
                    // 處理一般字符輸入
                    if (keyInfo.KeyChar != '\0')
                    {
                        var currentLine = Context.Texts[Context.Y];
                        
                        // 獲取當前文本
                        string currentText = new string(currentLine.Chars.Select(c => c.Value).ToArray());
                        
                        // 在光標位置插入字符
                        string newText = currentText.Insert(Context.X, keyInfo.KeyChar.ToString());
                        
                        // 更新文本
                        currentLine.SetText(0, newText);
                        
                        // 移動光標
                        Context.X++;
                    }
                    break;
            }
            
            // 重新渲染
            _render.Render(new RenderArgs
            {
                X = 0,
                Y = Context.Y,
                Text = Context.Texts[Context.Y]
            });
            
            // 設置光標位置
            Console.SetCursorPosition(Context.X, Context.Y);
        }
    }
}