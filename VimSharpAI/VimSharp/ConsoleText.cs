namespace VimSharp
{
    public class ConsoleText
    {
        public ColoredChar[] Chars { get; set; }
        public int Width => Chars.Length;

        public ConsoleText(string text)
        {
            Chars = text.Select(c => new ColoredChar(c, ConsoleColor.White, ConsoleColor.Black)).ToArray();
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