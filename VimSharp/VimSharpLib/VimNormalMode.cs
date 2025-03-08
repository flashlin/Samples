namespace VimSharpLib;
using System.Text;
using System.Linq;

public class VimNormalMode
{
    public required VimEditor Instance { get; set; }
    private bool _continueEditing = true;
    private ConsoleRender _render = new();
    
    public void WaitForInput()
    {
        while (_continueEditing)
        {
            var keyInfo = Console.ReadKey(intercept: true);
            
            // 確保當前行存在
            if (Instance.Context.Texts.Count <= Instance.Context.Y)
            {
                Instance.Context.Texts.Add(new ConsoleText());
            }
            
            switch (keyInfo.Key)
            {
                case ConsoleKey.Escape:
                    _continueEditing = false;
                    break;
                    
                case ConsoleKey.Backspace:
                    if (Instance.Context.X > 0)
                    {
                        // 獲取當前行
                        var currentLine = Instance.Context.Texts[Instance.Context.Y];
                        
                        // 獲取當前文本
                        string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
                        
                        // 計算實際索引位置
                        int actualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.X);
                        
                        if (actualIndex > 0)
                        {
                            // 獲取要刪除的字符
                            char charToDelete = currentText[actualIndex - 1];
                            
                            // 刪除字符
                            string newText = currentText.Remove(actualIndex - 1, 1);
                            
                            // 更新文本
                            currentLine.SetText(0, newText);
                            
                            // 移動光標（考慮中文字符寬度）
                            Instance.Context.X -= charToDelete.GetCharWidth();
                            
                            // 清除屏幕並重新渲染整行（對於 Backspace，我們需要重新渲染整行）
                            Console.Clear();
                            _render.Render(new RenderArgs
                            {
                                X = 0,
                                Y = Instance.Context.Y,
                                Text = Instance.Context.Texts[Instance.Context.Y]
                            });
                        }
                    }
                    break;
                    
                case ConsoleKey.LeftArrow:
                    if (Instance.Context.X > 0)
                    {
                        // 獲取當前文本
                        var currentLine = Instance.Context.Texts[Instance.Context.Y];
                        string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
                        
                        // 計算實際索引位置
                        int actualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.X);
                        
                        if (actualIndex > 0)
                        {
                            // 獲取前一個字符的寬度
                            char prevChar = currentText[actualIndex - 1];
                            Instance.Context.X -= prevChar.GetCharWidth();
                        }
                    }
                    break;
                    
                case ConsoleKey.RightArrow:
                    var currentLineForRight = Instance.Context.Texts[Instance.Context.Y];
                    string textForRight = new string(currentLineForRight.Chars.Select(c => c.Char).ToArray());
                    
                    // 計算實際索引位置
                    int actualIndexForRight = textForRight.GetStringIndexFromDisplayPosition(Instance.Context.X);
                    
                    if (actualIndexForRight < textForRight.Length)
                    {
                        // 獲取當前字符的寬度
                        char currentChar = textForRight[actualIndexForRight];
                        Instance.Context.X += currentChar.GetCharWidth();
                    }
                    break;
                    
                case ConsoleKey.UpArrow:
                    if (Instance.Context.Y > 0)
                    {
                        Instance.Context.Y--;
                        // 確保 X 不超過新行的顯示寬度
                        string upLineText = new string(Instance.Context.Texts[Instance.Context.Y].Chars.Select(c => c.Char).ToArray());
                        int upLineWidth = upLineText.GetStringDisplayWidth();
                        if (Instance.Context.X > upLineWidth)
                        {
                            Instance.Context.X = upLineWidth;
                        }
                        
                        // 清除屏幕並重新渲染當前行
                        Console.Clear();
                        _render.Render(new RenderArgs
                        {
                            X = 0,
                            Y = Instance.Context.Y,
                            Text = Instance.Context.Texts[Instance.Context.Y]
                        });
                    }
                    break;
                    
                case ConsoleKey.DownArrow:
                    if (Instance.Context.Y < Instance.Context.Texts.Count - 1)
                    {
                        Instance.Context.Y++;
                        // 確保 X 不超過新行的顯示寬度
                        string downLineText = new string(Instance.Context.Texts[Instance.Context.Y].Chars.Select(c => c.Char).ToArray());
                        int downLineWidth = downLineText.GetStringDisplayWidth();
                        if (Instance.Context.X > downLineWidth)
                        {
                            Instance.Context.X = downLineWidth;
                        }
                        
                        // 清除屏幕並重新渲染當前行
                        Console.Clear();
                        _render.Render(new RenderArgs
                        {
                            X = 0,
                            Y = Instance.Context.Y,
                            Text = Instance.Context.Texts[Instance.Context.Y]
                        });
                    }
                    break;
                    
                default:
                    // 處理一般字符輸入
                    if (keyInfo.KeyChar != '\0')
                    {
                        var currentLine = Instance.Context.Texts[Instance.Context.Y];
                        
                        // 獲取當前文本
                        string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());
                        
                        // 計算實際索引位置
                        int actualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.X);
                        
                        // 在實際索引位置插入字符
                        string newText = currentText.Insert(actualIndex, keyInfo.KeyChar.ToString());
                        
                        // 更新文本
                        currentLine.SetText(0, newText);
                        
                        // 移動光標（考慮中文字符寬度）
                        Instance.Context.X += keyInfo.KeyChar.GetCharWidth();
                    }
                    break;
            }
            
            _render.Render(new RenderArgs()
            {
                X = 0, 
                Y = Instance.Context.Y, 
                Text = Instance.Context.Texts[Instance.Context.Y]
            });
            
            // 不需要重新渲染整行，只需設置光標位置
            Console.SetCursorPosition(Instance.Context.X, Instance.Context.Y);
        }
    }
} 