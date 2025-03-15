using System.Text;

namespace VimSharpLib
{
    public class VimEditor
    {
        public List<ConsoleText> Texts { get; private set; } = new List<ConsoleText>();
        public int CursorX { get; set; } = 0;
        public int CursorY { get; set; } = 0;
        public int OffsetX { get; set; } = 0;
        public int OffsetY { get; set; } = 0;
        public ViewArea ViewPort { get; set; }
        public bool IsStatusBarVisible { get; set; } = true;
        public ConsoleText StatusBar { get; set; }
        public bool IsRelativeNumberVisible { get; set; } = true;
        public IConsoleDevice Console { get; private set; }
        public IVimMode Mode { get; set; }

        public VimEditor(IConsoleDevice consoleDevice)
        {
            Console = consoleDevice;
            ViewPort = new ViewArea(0, 0, Console.WindowWidth, Console.WindowHeight);
            StatusBar = new ConsoleText($"-- NORMAL --");
            
            // 添加一個空行，以確保至少有一行內容
            Texts.Add(new ConsoleText(""));
        }

        public void OpenFile(string file)
        {
            Texts.Clear();
            if (File.Exists(file))
            {
                foreach (var line in File.ReadAllLines(file))
                {
                    Texts.Add(new ConsoleText(line));
                }
            }
            else
            {
                Texts.Add(new ConsoleText(""));
            }
            
            CursorX = 0;
            CursorY = 0;
            OffsetX = 0;
            OffsetY = 0;
        }

        public void OpenText(string text)
        {
            Texts.Clear();
            foreach (var line in text.Split('\n'))
            {
                // 移除每行末尾的 \r 字符
                string cleanLine = line.TrimEnd('\r');
                Texts.Add(new ConsoleText(cleanLine));
            }
            CursorX = 0;
            CursorY = 0;
            OffsetX = 0;
            OffsetY = 0;
        }

        public void Render(ColoredChar[,]? screenBuffer = null)
        {
            int width = Console.WindowWidth;
            int height = Console.WindowHeight;
            
            if (screenBuffer == null)
            {
                screenBuffer = new ColoredChar[width, height];
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        screenBuffer[x, y] = ColoredChar.Empty;
                    }
                }
            }

            // 根據 ViewPort 繪製 Texts 內容
            int relativeNumberWidth = IsRelativeNumberVisible ? 4 : 0;
            int textStartX = ViewPort.X + relativeNumberWidth;
            
            for (int y = 0; y < ViewPort.Height; y++)
            {
                int textY = y + OffsetY;
                if (textY >= 0 && textY < Texts.Count)
                {
                    // 如果啟用了相對行號，則繪製行號區域
                    if (IsRelativeNumberVisible)
                    {
                        int lineNumber = textY;
                        string lineNumberText = lineNumber.ToString().PadLeft(3);
                        for (int nx = 0; nx < lineNumberText.Length; nx++)
                        {
                            if (nx < relativeNumberWidth && ViewPort.X + nx < width)
                            {
                                screenBuffer[ViewPort.X + nx, ViewPort.Y + y] = new ColoredChar(
                                    lineNumberText[nx],
                                    ConsoleColor.Gray,
                                    ConsoleColor.Black
                                );
                            }
                        }
                    }

                    // 繪製文本內容
                    var textLine = Texts[textY];
                    for (int x = 0; x < ViewPort.Width - relativeNumberWidth; x++)
                    {
                        int textX = x + OffsetX;
                        if (textX >= 0 && textX < textLine.Width && textStartX + x < width)
                        {
                            screenBuffer[textStartX + x, ViewPort.Y + y] = textLine.Chars[textX];
                        }
                        else if (textStartX + x < width)
                        {
                            screenBuffer[textStartX + x, ViewPort.Y + y] = ColoredChar.ViewEmpty;
                        }
                    }
                }
                else if (ViewPort.Y + y < height)
                {
                    // 繪製空視圖區域
                    for (int x = 0; x < ViewPort.Width; x++)
                    {
                        if (ViewPort.X + x < width)
                        {
                            screenBuffer[ViewPort.X + x, ViewPort.Y + y] = ColoredChar.ViewEmpty;
                        }
                    }
                }
            }

            // 繪製狀態列
            if (IsStatusBarVisible && ViewPort.Y + ViewPort.Height - 1 < height)
            {
                int statusY = ViewPort.Y + ViewPort.Height - 1;
                for (int x = 0; x < ViewPort.Width && x < StatusBar.Width; x++)
                {
                    if (ViewPort.X + x < width)
                    {
                        screenBuffer[ViewPort.X + x, statusY] = StatusBar.Chars[x];
                    }
                }
            }

            WriteToConsole(screenBuffer);
            
            // 設置游標位置
            int cursorPosX = textStartX + CursorX - OffsetX;
            int cursorPosY = ViewPort.Y + CursorY - OffsetY;
            Console.SetCursorPosition(cursorPosX, cursorPosY);
        }

        public void WriteToConsole(ColoredChar[,]? screenBuffer)
        {
            if (screenBuffer == null) return;
            
            // 使用 IConsoleDevice 隱藏游標
            // 由於 IConsoleDevice 沒有直接控制游標可見性的方法，我們需要添加一個 ANSI 轉義序列來隱藏游標
            Console.Write("\u001b[?25l"); // 隱藏游標的 ANSI 轉義序列
            
            int width = screenBuffer.GetLength(0);
            int height = screenBuffer.GetLength(1);
            
            // 使用 StringBuilder 收集所有輸出
            var sb = new StringBuilder();
            
            for (int y = 0; y < height; y++)
            {
                // 使用 ANSI 轉義序列設置游標位置
                sb.Append($"\u001b[{y + 1};1H"); // 將游標移動到 y+1 行，第 1 列
                
                for (int x = 0; x < width; x++)
                {
                    var coloredChar = screenBuffer[x, y];
                    sb.Append(coloredChar.ToAnsiString());
                }
            }
            
            // 添加重置顏色的 ANSI 轉義序列
            sb.Append("\u001b[0m");
            
            // 一次性輸出所有內容
            Console.Write(sb.ToString());
            
            // 使用 ANSI 轉義序列顯示游標
            Console.Write("\u001b[?25h"); // 顯示游標的 ANSI 轉義序列
        }

        // 游標移動方法
        public void MoveCursorLeft()
        {
            if (CursorX > 0)
            {
                CursorX--;
            }
            else if (CursorY > 0)
            {
                CursorY--;
                CursorX = Math.Max(0, Texts[GetActualTextY()].Width - 1);
            }

            AdjustViewPortOffset();
        }

        public void MoveCursorRight()
        {
            int actualY = GetActualTextY();
            if (actualY < Texts.Count)
            {
                int lineWidth = Texts[actualY].Width;
                if (CursorX < lineWidth - 1)
                {
                    CursorX++;
                }
                else if (actualY < Texts.Count - 1)
                {
                    CursorY++;
                    CursorX = 0;
                }
            }

            AdjustViewPortOffset();
        }

        public void MoveCursorUp()
        {
            if (CursorY > 0)
            {
                CursorY--;
                int lineWidth = Texts[GetActualTextY()].Width;
                CursorX = Math.Min(CursorX, Math.Max(0, lineWidth - 1));
            }

            AdjustViewPortOffset();
        }

        public void MoveCursorDown()
        {
            if (CursorY < Texts.Count - 1)
            {
                CursorY++;
                int lineWidth = Texts[GetActualTextY()].Width;
                CursorX = Math.Min(CursorX, Math.Max(0, lineWidth - 1));
            }

            AdjustViewPortOffset();
        }

        public void MoveCursorToEndOfLine()
        {
            int actualY = GetActualTextY();
            if (actualY < Texts.Count)
            {
                // 獲取當前行的文本
                var text = Texts[actualY];
                
                if (text.Width > 0)
                {
                    // 檢查最後一個字符是否為中文字符（ASCII > 127）
                    if (text.Chars[text.Width - 1].Char > 127)
                    {
                        // 如果最後一個字符是中文字符，使用 Width - 2
                        CursorX = Math.Max(0, text.Width - 2);
                    }
                    else
                    {
                        // 如果最後一個字符不是中文字符，使用 Width - 1
                        CursorX = Math.Max(0, text.Width - 1);
                    }
                }
                else
                {
                    // 空行，將游標設為 0
                    CursorX = 0;
                }
                
                AdjustViewPortOffset();
            }
        }

        public void MoveCursorToStartOfLine()
        {
            CursorX = 0;
            AdjustViewPortOffset();
        }

        /// <summary>
        /// 設置視口區域
        /// </summary>
        /// <param name="x">視口的 X 座標</param>
        /// <param name="y">視口的 Y 座標</param>
        /// <param name="width">視口的寬度</param>
        /// <param name="height">視口的高度</param>
        public void SetViewPort(int x, int y, int width, int height)
        {
            ViewPort = new ViewArea(x, y, width, height);
            AdjustViewPortOffset(); // 調整偏移量以確保游標在可視範圍內
        }

        // 獲取實際的文本坐標
        public int GetActualTextX()
        {
            return CursorX;
        }

        public int GetActualTextY()
        {
            return CursorY;
        }

        private void AdjustViewPortOffset()
        {
            int relativeNumberWidth = IsRelativeNumberVisible ? 4 : 0;
            int effectiveWidth = ViewPort.Width - relativeNumberWidth;
            
            // 調整 X 偏移量
            if (CursorX < OffsetX)
            {
                OffsetX = CursorX;
            }
            else if (CursorX >= OffsetX + effectiveWidth)
            {
                OffsetX = CursorX - effectiveWidth + 1;
            }

            // 調整 Y 偏移量
            if (CursorY < OffsetY)
            {
                OffsetY = CursorY;
            }
            else if (CursorY >= OffsetY + ViewPort.Height)
            {
                OffsetY = CursorY - ViewPort.Height + 1;
            }
        }
    }
} 
