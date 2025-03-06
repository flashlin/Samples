namespace VimSharpLib;

public class ConsoleText
{
    public ConsoleCharacter[] Chars { get; set; } = [];
    public int Width
    {
        get => Chars.Length;
        set
        {
            if (Chars.Length < value)
            {
                var newChars = new ConsoleCharacter[value];
                Array.Copy(Chars, newChars, Chars.Length);
                for (var i = Chars.Length; i < value; i++)
                {
                    newChars[i] = ConsoleCharacter.Empty;
                }
                Chars = newChars;
                return;
            }
            if (Chars.Length > value)
            {
                var newChars = new ConsoleCharacter[value];
                Array.Copy(Chars, newChars, value);
                Chars = newChars;
            }
        }
    }

    public void SetText(int x, string text)
    {
        Width = text.Length + x;
        for (var i = 0; i < text.Length; i++)
        {
            Chars[x + i] = new ConsoleCharacter()
            {
                Value = text[i],
                Color = ConsoleColor.Black,
                BackgroundColor = ConsoleColor.Cyan
            };
        }
    }
}