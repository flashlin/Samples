namespace VimSharpLib;
using System.Text;
using System.Linq;

public class VimNormalMode : IVimMode
{
    public required VimEditor Instance { get; set; }

    private bool _continueEditing = true;

    public void WaitForInput()
    {
        while (_continueEditing)
        {
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
                    }
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
                    }
                    break;
            }

            // 渲染當前行
            Instance.Render();
        }
    }
}