namespace T1.ParserKit;

public class Token
{
    public static readonly Token Empty = new Token
    {
        Text = string.Empty,
        Index = -1
    };

    public string Text { get; set; } = string.Empty;
    public int Index { get; set; }
}



