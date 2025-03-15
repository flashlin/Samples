namespace VimSharpLib
{
    public class ConsoleText
    {
        public ColoredChar[] Chars { get; set; }
        public int Width => Chars.Length;

        public ConsoleText(string text)
        {
            // 計算實際需要的數組大小：普通字符 + 中文字符（每個中文字符需要額外一個空間存放 '\0'）
            int chineseCharsCount = text.Count(c => c > 127);
            int totalLength = text.Length + chineseCharsCount;
            
            Chars = new ColoredChar[totalLength];
            
            int index = 0;
            foreach (char c in text)
            {
                // 對於空白字符，使用 ColoredChar.Empty
                if (c == ' ')
                {
                    Chars[index++] = ColoredChar.ViewEmpty;
                }
                // 對於中文字符（ASCII 值大於 127 的字符），添加一個 '\0' 字符
                else if (c > 127)
                {
                    Chars[index++] = new ColoredChar(c, ConsoleColor.White, ConsoleColor.DarkGray);
                    Chars[index++] = new ColoredChar('\0', ConsoleColor.White, ConsoleColor.DarkGray);
                }
                else
                {
                    Chars[index++] = new ColoredChar(c, ConsoleColor.White, ConsoleColor.DarkGray);
                }
            }
        }

        public ConsoleText(ColoredChar[] chars)
        {
            Chars = chars;
        }

        public ConsoleText(int size)
        {
            Chars = new ColoredChar[size];
            for (int i = 0; i < size; i++)
            {
                Chars[i] = ColoredChar.Empty;
            }
        }
    }
} 
