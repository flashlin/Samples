namespace VimSharpLib;
using System.Text;

public class VimEditEditor
{
    ConsoleRender _render { get; set; } = new();
    bool _continueEditing = true;
    public ConsoleContext Context { get; set; } = new();
    
    // 添加一個編碼轉換器用於檢測中文字符
    private static readonly Encoding Big5Encoding = Encoding.GetEncoding("big5");
    
    public void Run()
    {
        // 初始渲染
        Console.Clear();
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
        
        while (_continueEditing)
        {
            WaitForInput();
        }
    }
    
    // 檢查字符是否為中文字符（在 Big5 編碼中佔用 2 個字節）
    private bool IsChinese(char c)
    {
        // 使用 Big5 編碼檢查字符的字節長度
        var bytes = Encoding.ASCII.GetBytes([c]);
        return bytes.Length > 1;
    }
    
    // 獲取字符在 Big5 編碼中的寬度（中文為 2，其他為 1）
    private int GetCharWidth(char c)
    {
        return IsChinese(c) ? 2 : 1;
    }
    
    // 計算字符串中的顯示寬度（考慮中文字符佔 2 個位置）
    private int GetStringDisplayWidth(string text)
    {
        int width = 0;
        foreach (char c in text)
        {
            width += GetCharWidth(c);
        }
        return width;
    }
    
    // 根據顯示位置獲取字符串中的實際索引
    private int GetStringIndexFromDisplayPosition(string text, int displayPosition)
    {
        int currentWidth = 0;
        for (int i = 0; i < text.Length; i++)
        {
            if (currentWidth >= displayPosition)
                return i;
                
            currentWidth += GetCharWidth(text[i]);
        }
        return text.Length;
    }
    
    public void WaitForInput()
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
                _continueEditing = false;
                break;
                    
            case ConsoleKey.Backspace:
                if (Context.X > 0)
                {
                    // 獲取當前行
                    var currentLine = Context.Texts[Context.Y];
                        
                    // 獲取當前文本
                    string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
                        
                    // 計算實際索引位置
                    int actualIndex = GetStringIndexFromDisplayPosition(currentText, Context.X);
                        
                    if (actualIndex > 0)
                    {
                        // 獲取要刪除的字符
                        char charToDelete = currentText[actualIndex - 1];
                            
                        // 刪除字符
                        string newText = currentText.Remove(actualIndex - 1, 1);
                            
                        // 更新文本
                        currentLine.SetText(0, newText);
                            
                        // 移動光標（考慮中文字符寬度）
                        Context.X -= GetCharWidth(charToDelete);
                        
                        // 清除屏幕並重新渲染整行（對於 Backspace，我們需要重新渲染整行）
                        Console.Clear();
                        _render.Render(new RenderArgs
                        {
                            X = 0,
                            Y = Context.Y,
                            Text = Context.Texts[Context.Y]
                        });
                    }
                }
                break;
                    
            case ConsoleKey.LeftArrow:
                if (Context.X > 0)
                {
                    // 獲取當前文本
                    var currentLine = Context.Texts[Context.Y];
                    string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
                        
                    // 計算實際索引位置
                    int actualIndex = GetStringIndexFromDisplayPosition(currentText, Context.X);
                        
                    if (actualIndex > 0)
                    {
                        // 獲取前一個字符的寬度
                        char prevChar = currentText[actualIndex - 1];
                        Context.X -= GetCharWidth(prevChar);
                    }
                }
                break;
                    
            case ConsoleKey.RightArrow:
                var currentLineForRight = Context.Texts[Context.Y];
                string textForRight = new string(currentLineForRight.Chars.Select(c => c.Char).ToArray());
                    
                // 計算實際索引位置
                int actualIndexForRight = GetStringIndexFromDisplayPosition(textForRight, Context.X);
                    
                if (actualIndexForRight < textForRight.Length)
                {
                    // 獲取當前字符的寬度
                    char currentChar = textForRight[actualIndexForRight];
                    Context.X += GetCharWidth(currentChar);
                }
                break;
                    
            case ConsoleKey.UpArrow:
                if (Context.Y > 0)
                {
                    Context.Y--;
                    // 確保 X 不超過新行的顯示寬度
                    string upLineText = new string(Context.Texts[Context.Y].Chars.Select(c => c.Char).ToArray());
                    int upLineWidth = GetStringDisplayWidth(upLineText);
                    if (Context.X > upLineWidth)
                    {
                        Context.X = upLineWidth;
                    }
                    
                    // 清除屏幕並重新渲染當前行
                    Console.Clear();
                    _render.Render(new RenderArgs
                    {
                        X = 0,
                        Y = Context.Y,
                        Text = Context.Texts[Context.Y]
                    });
                }
                break;
                    
            case ConsoleKey.DownArrow:
                if (Context.Y < Context.Texts.Count - 1)
                {
                    Context.Y++;
                    // 確保 X 不超過新行的顯示寬度
                    string downLineText = new string(Context.Texts[Context.Y].Chars.Select(c => c.Char).ToArray());
                    int downLineWidth = GetStringDisplayWidth(downLineText);
                    if (Context.X > downLineWidth)
                    {
                        Context.X = downLineWidth;
                    }
                    
                    // 清除屏幕並重新渲染當前行
                    Console.Clear();
                    _render.Render(new RenderArgs
                    {
                        X = 0,
                        Y = Context.Y,
                        Text = Context.Texts[Context.Y]
                    });
                }
                break;
                    
            default:
                // 處理一般字符輸入
                if (keyInfo.KeyChar != '\0')
                {
                    var currentLine = Context.Texts[Context.Y];
                        
                    // 獲取當前文本
                    string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
                        
                    // 計算實際索引位置
                    int actualIndex = GetStringIndexFromDisplayPosition(currentText, Context.X);
                        
                    // 在實際索引位置插入字符
                    string newText = currentText.Insert(actualIndex, keyInfo.KeyChar.ToString());
                        
                    // 更新文本
                    currentLine.SetText(0, newText);
                    
                    // 只渲染新添加的字符
                    //_render.RenderChar(Context.X, Context.Y, currentLine.Chars[actualIndex]);
                        
                    // 移動光標（考慮中文字符寬度）
                    Context.X += GetCharWidth(keyInfo.KeyChar);
                }
                break;
        }
        
        _render.Render(new RenderArgs()
        {
            X = 0, 
            Y = Context.Y, 
            Text = Context.Texts[Context.Y]
        });
            
        // 不需要重新渲染整行，只需設置光標位置
        Console.SetCursorPosition(Context.X, Context.Y);
    }
}