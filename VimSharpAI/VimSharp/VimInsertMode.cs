namespace VimSharp
{
    public class VimInsertMode : IVimMode
    {
        public VimEditor Instance { get; set; }

        public void WaitForInput()
        {
            var keyInfo = Instance.Console.ReadKey(intercept: true);

            // 按 ESC 鍵返回普通模式
            if (keyInfo.Key == ConsoleKey.Escape)
            {
                Instance.StatusBar = new ConsoleText("-- NORMAL --");
                Instance.Mode = new VimNormalMode { Instance = Instance };
                return;
            }

            // 處理特殊按鍵
            switch (keyInfo.Key)
            {
                case ConsoleKey.Backspace:
                    HandleBackspace();
                    break;
                case ConsoleKey.Enter:
                    HandleEnter();
                    break;
                case ConsoleKey.LeftArrow:
                    Instance.MoveCursorLeft();
                    break;
                case ConsoleKey.RightArrow:
                    Instance.MoveCursorRight();
                    break;
                case ConsoleKey.UpArrow:
                    Instance.MoveCursorUp();
                    break;
                case ConsoleKey.DownArrow:
                    Instance.MoveCursorDown();
                    break;
                default:
                    // 處理一般文字輸入
                    if (keyInfo.KeyChar >= 32 && keyInfo.KeyChar <= 126)
                    {
                        InsertCharacter(keyInfo.KeyChar);
                    }
                    break;
            }
        }

        private void HandleBackspace()
        {
            int x = Instance.GetActualTextX();
            int y = Instance.GetActualTextY();

            if (x > 0)
            {
                // 刪除當前行中的字符
                var line = Instance.Texts[y];
                var newChars = new ColoredChar[line.Width - 1];
                
                Array.Copy(line.Chars, 0, newChars, 0, x - 1);
                if (x < line.Width)
                {
                    Array.Copy(line.Chars, x, newChars, x - 1, line.Width - x);
                }
                
                Instance.Texts[y] = new ConsoleText(newChars);
                Instance.MoveCursorLeft();
            }
            else if (y > 0)
            {
                // 合併兩行
                var previousLine = Instance.Texts[y - 1];
                var currentLine = Instance.Texts[y];
                
                var newChars = new ColoredChar[previousLine.Width + currentLine.Width];
                Array.Copy(previousLine.Chars, 0, newChars, 0, previousLine.Width);
                Array.Copy(currentLine.Chars, 0, newChars, previousLine.Width, currentLine.Width);
                
                Instance.Texts[y - 1] = new ConsoleText(newChars);
                Instance.Texts.RemoveAt(y);
                
                Instance.CursorY = y - 1;
                Instance.CursorX = previousLine.Width;
            }
        }

        private void HandleEnter()
        {
            int x = Instance.GetActualTextX();
            int y = Instance.GetActualTextY();
            
            var currentLine = Instance.Texts[y];
            
            // 創建新行
            if (x == 0)
            {
                // 在當前行前插入空行
                Instance.Texts.Insert(y, new ConsoleText(""));
                Instance.CursorY++;
            }
            else if (x >= currentLine.Width)
            {
                // 在當前行後插入空行
                Instance.Texts.Insert(y + 1, new ConsoleText(""));
                Instance.CursorY++;
                Instance.CursorX = 0;
            }
            else
            {
                // 拆分當前行
                var firstPart = new ColoredChar[x];
                var secondPart = new ColoredChar[currentLine.Width - x];
                
                Array.Copy(currentLine.Chars, 0, firstPart, 0, x);
                Array.Copy(currentLine.Chars, x, secondPart, 0, currentLine.Width - x);
                
                Instance.Texts[y] = new ConsoleText(firstPart);
                Instance.Texts.Insert(y + 1, new ConsoleText(secondPart));
                
                Instance.CursorY++;
                Instance.CursorX = 0;
            }
        }

        private void InsertCharacter(char ch)
        {
            int x = Instance.GetActualTextX();
            int y = Instance.GetActualTextY();
            
            var line = Instance.Texts[y];
            var newChars = new ColoredChar[line.Width + 1];
            
            // 複製插入點之前的內容
            if (x > 0)
            {
                Array.Copy(line.Chars, 0, newChars, 0, x);
            }
            
            // 插入新字符
            newChars[x] = new ColoredChar(ch, ConsoleColor.White, ConsoleColor.Black);
            
            // 複製插入點之後的內容
            if (x < line.Width)
            {
                Array.Copy(line.Chars, x, newChars, x + 1, line.Width - x);
            }
            
            Instance.Texts[y] = new ConsoleText(newChars);
            Instance.MoveCursorRight();
        }
    }
} 