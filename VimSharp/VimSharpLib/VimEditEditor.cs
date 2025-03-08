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
}