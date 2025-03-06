namespace VimSharpLib;

public class VimEditor
{
    ConsoleRender _render { get; set; } = new();
    public ConsoleContext Context { get; set; } = new();

    public void Initialize()
    {
    }

    public void Render()
    {
        Context.SetText(0, 0, "Hello, World!");
        _render.Render(new RenderArgs
        {
            X = 0,
            Y = 0,
            Text = Context.Texts[0]
        });
    }
}

public class ConsoleContext
{
    public List<ConsoleText> Texts { get; set; } = [];

    public void SetText(int x, int y, string text)
    {
        if (Texts.Count <= y)
        {
            Texts.Add(new ConsoleText());
        }
        var consoleText = Texts[y];
        consoleText.SetText(x, text);
    }
}

public class ConsoleRender
{
    public void Render(RenderArgs args)
    {
        Console.SetCursorPosition(args.X, args.Y);
        foreach (var c in args.Text.Chars)
        {
            Console.ForegroundColor = c.Color;
            Console.BackgroundColor = c.BackgroundColor;
            Console.Write(c.Value);
        }

        Console.ResetColor();
    }
}

public class ConsoleCharacter
{
    public static ConsoleCharacter Empty = new ConsoleCharacter
    {
        Value = '\0',
        Color = ConsoleColor.Black,
        BackgroundColor = ConsoleColor.Black
    };
    public char Value { get; set; }
    public ConsoleColor Color { get; set; }
    public ConsoleColor BackgroundColor { get; set; }
}

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
                Color = ConsoleColor.White,
                BackgroundColor = ConsoleColor.Black
            };
        }
    }
}

public class RenderArgs
{
    public int X { get; set; }
    public int Y { get; set; }
    public ConsoleText Text { get; set; }
}