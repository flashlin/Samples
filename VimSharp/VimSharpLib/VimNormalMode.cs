namespace VimSharpLib;
using System.Text;
using System.Linq;

public class VimNormalMode : IVimMode
{
    public required VimEditor Instance { get; set; }
    
    /// <summary>
    /// 檢查並調整游標位置和偏移量，確保游標在可見區域內
    /// </summary>
    private void AdjustCursorAndOffset()
    {
        // 計算游標在屏幕上的位置
        int cursorScreenX = Instance.Context.CursorX - Instance.Context.OffsetX;
        int cursorScreenY = Instance.Context.CursorY - Instance.Context.OffsetY;

        // 檢查游標是否超出右邊界
        if (cursorScreenX >= Instance.Context.ViewPort.Width)
        {
            // 調整水平偏移量，使游標位於可見區域的右邊界
            Instance.Context.OffsetX = Instance.Context.CursorX - Instance.Context.ViewPort.Width + 1;
        }
        // 檢查游標是否超出左邊界
        else if (cursorScreenX < 0)
        {
            // 調整水平偏移量，使游標位於可見區域的左邊界
            Instance.Context.OffsetX = Instance.Context.CursorX;
        }

        // 檢查游標是否超出下邊界
        if (cursorScreenY >= Instance.Context.ViewPort.Height)
        {
            // 調整垂直偏移量，使游標位於可見區域的下邊界
            Instance.Context.OffsetY = Instance.Context.CursorY - Instance.Context.ViewPort.Height + 1;
        }
        // 檢查游標是否超出上邊界
        else if (cursorScreenY < 0)
        {
            // 調整垂直偏移量，使游標位於可見區域的上邊界
            Instance.Context.OffsetY = Instance.Context.CursorY;
        }
    }

    public void WaitForInput()
    {
        // 設置為垂直線游標 (DECSCUSR 6)
        Console.Write("\x1b[6 q");
        
        var keyInfo = Console.ReadKey(intercept: true);

        // 確保當前行存在
        if (Instance.Context.Texts.Count <= Instance.Context.CursorY)
        {
            Instance.Context.Texts.Add(new ConsoleText());
        }

        switch (keyInfo.Key)
        {
            case ConsoleKey.Escape:
                Instance.Mode = new VimVisualMode { Instance = Instance };
                break;

            case ConsoleKey.Backspace:
                if (Instance.Context.CursorX > 0)
                {
                    // 獲取當前行
                    var currentLine = Instance.Context.Texts[Instance.Context.CursorY];

                    // 獲取當前文本
                    string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());

                    // 計算實際索引位置
                    int actualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);

                    if (actualIndex > 0)
                    {
                        // 獲取要刪除的字符
                        char charToDelete = currentText[actualIndex - 1];

                        // 刪除字符
                        string newText = currentText.Remove(actualIndex - 1, 1);

                        // 更新文本
                        currentLine.SetText(0, newText);

                        // 移動光標（考慮中文字符寬度）
                        Instance.Context.CursorX -= charToDelete.GetCharWidth();
                    }
                }
                break;

            case ConsoleKey.LeftArrow:
                if (Instance.Context.CursorX > 0)
                {
                    // 獲取當前文本
                    var currentLine = Instance.Context.Texts[Instance.Context.CursorY];
                    string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());

                    // 計算實際索引位置
                    int actualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);

                    if (actualIndex > 0)
                    {
                        // 獲取前一個字符的寬度
                        char prevChar = currentText[actualIndex - 1];
                        Instance.Context.CursorX -= prevChar.GetCharWidth();
                        // 檢查並調整游標位置和偏移量
                        AdjustCursorAndOffset();
                    }
                }
                break;

            case ConsoleKey.RightArrow:
                var currentLineForRight = Instance.Context.Texts[Instance.Context.CursorY];
                string textForRight = new string(currentLineForRight.Chars.Select(c => c.Char).ToArray());

                // 計算實際索引位置
                int actualIndexForRight = textForRight.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);

                if (actualIndexForRight < textForRight.Length)
                {
                    // 獲取當前字符的寬度
                    char currentChar = textForRight[actualIndexForRight];
                    Instance.Context.CursorX += currentChar.GetCharWidth();
                    // 檢查並調整游標位置和偏移量
                    AdjustCursorAndOffset();
                }
                break;

            case ConsoleKey.UpArrow:
                if (Instance.Context.CursorY > 0)
                {
                    Instance.Context.CursorY--;
                    // 確保 X 不超過新行的顯示寬度
                    string upLineText = new string(Instance.Context.Texts[Instance.Context.CursorY].Chars.Select(c => c.Char).ToArray());
                    int upLineWidth = upLineText.GetStringDisplayWidth();
                    if (Instance.Context.CursorX > upLineWidth)
                    {
                        Instance.Context.CursorX = upLineWidth;
                    }
                    // 檢查並調整游標位置和偏移量
                    AdjustCursorAndOffset();
                }
                break;

            case ConsoleKey.DownArrow:
                if (Instance.Context.CursorY < Instance.Context.Texts.Count - 1)
                {
                    Instance.Context.CursorY++;
                    // 確保 X 不超過新行的顯示寬度
                    string downLineText = new string(Instance.Context.Texts[Instance.Context.CursorY].Chars.Select(c => c.Char).ToArray());
                    int downLineWidth = downLineText.GetStringDisplayWidth();
                    if (Instance.Context.CursorX > downLineWidth)
                    {
                        Instance.Context.CursorX = downLineWidth;
                    }
                    // 檢查並調整游標位置和偏移量
                    AdjustCursorAndOffset();
                }
                break;

            case ConsoleKey.Enter:
                // 在當前行後插入新行
                Instance.Context.CursorY++;
                Instance.Context.CursorX = 0;
                
                // 確保新行存在
                if (Instance.Context.Texts.Count <= Instance.Context.CursorY)
                {
                    Instance.Context.Texts.Add(new ConsoleText());
                }
                
                // 檢查並調整游標位置和偏移量
                AdjustCursorAndOffset();
                break;

            default:
                // 處理一般字符輸入
                if (char.IsLetterOrDigit(keyInfo.KeyChar) || char.IsPunctuation(keyInfo.KeyChar) || char.IsWhiteSpace(keyInfo.KeyChar))
                {
                    var currentLine = Instance.Context.Texts[Instance.Context.CursorY];

                    // 獲取當前文本
                    string currentText = new string(currentLine.Chars.Select(c => c.Char).ToArray());

                    // 計算實際索引位置
                    int actualIndex = currentText.GetStringIndexFromDisplayPosition(Instance.Context.CursorX);

                    // 在實際索引位置插入字符
                    string newText = currentText.Insert(actualIndex, keyInfo.KeyChar.ToString());

                    // 更新文本
                    currentLine.SetText(0, newText);

                    // 移動光標（考慮中文字符寬度）
                    Instance.Context.CursorX += keyInfo.KeyChar.GetCharWidth();
                    // 檢查並調整游標位置和偏移量
                    AdjustCursorAndOffset();
                }
                break;
        }

        // 渲染當前行
        Instance.Render();
    }
}