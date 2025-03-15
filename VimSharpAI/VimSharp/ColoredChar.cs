namespace VimSharp
{
    public class ColoredChar
    {
        public char Char { get; set; }
        public ConsoleColor Foreground { get; set; }
        public ConsoleColor Background { get; set; }

        public ColoredChar(char ch, ConsoleColor foreground, ConsoleColor background)
        {
            Char = ch;
            Foreground = foreground;
            Background = background;
        }

        public static readonly ColoredChar ViewEmpty = new ColoredChar(' ', ConsoleColor.White, ConsoleColor.DarkGray);
        public static readonly ColoredChar Empty = new ColoredChar(' ', ConsoleColor.White, ConsoleColor.Black);
        
        // ANSI 顏色代碼映射表
        private static readonly Dictionary<ConsoleColor, string> ForegroundColors = new Dictionary<ConsoleColor, string>
        {
            { ConsoleColor.Black, "30" },
            { ConsoleColor.DarkRed, "31" },
            { ConsoleColor.DarkGreen, "32" },
            { ConsoleColor.DarkYellow, "33" },
            { ConsoleColor.DarkBlue, "34" },
            { ConsoleColor.DarkMagenta, "35" },
            { ConsoleColor.DarkCyan, "36" },
            { ConsoleColor.Gray, "37" },
            { ConsoleColor.DarkGray, "90" },
            { ConsoleColor.Red, "91" },
            { ConsoleColor.Green, "92" },
            { ConsoleColor.Yellow, "93" },
            { ConsoleColor.Blue, "94" },
            { ConsoleColor.Magenta, "95" },
            { ConsoleColor.Cyan, "96" },
            { ConsoleColor.White, "97" }
        };
        
        private static readonly Dictionary<ConsoleColor, string> BackgroundColors = new Dictionary<ConsoleColor, string>
        {
            { ConsoleColor.Black, "40" },
            { ConsoleColor.DarkRed, "41" },
            { ConsoleColor.DarkGreen, "42" },
            { ConsoleColor.DarkYellow, "43" },
            { ConsoleColor.DarkBlue, "44" },
            { ConsoleColor.DarkMagenta, "45" },
            { ConsoleColor.DarkCyan, "46" },
            { ConsoleColor.Gray, "47" },
            { ConsoleColor.DarkGray, "100" },
            { ConsoleColor.Red, "101" },
            { ConsoleColor.Green, "102" },
            { ConsoleColor.Yellow, "103" },
            { ConsoleColor.Blue, "104" },
            { ConsoleColor.Magenta, "105" },
            { ConsoleColor.Cyan, "106" },
            { ConsoleColor.White, "107" }
        };

        /// <summary>
        /// 將 ColoredChar 轉換為帶 ANSI 顏色代碼的字符串
        /// </summary>
        /// <returns>格式化的 ANSI 字符串</returns>
        public string ToAnsiString()
        {
            string foreCode = ForegroundColors.TryGetValue(Foreground, out var fore) ? fore : "37"; // 默認白色
            string backCode = BackgroundColors.TryGetValue(Background, out var back) ? back : "40"; // 默認黑色
            
            // 格式為 ESC[<背景色>;<前景色>m<字符>
            return $"\u001b[{backCode};{foreCode}m{Char}";
        }
    }
} 