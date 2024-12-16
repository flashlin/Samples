namespace T1.SqlSharp;

public class TextSpan
{
    public static TextSpan None { get; } = new();
    public string Word { get; set; } = string.Empty;
    public int Offset { get; set; } = -1;
    public int Length { get; set; }

    public static TextSpan Empty(int startPosition)
    {
        return new TextSpan()
        {
            Offset = startPosition,
            Length = 0
        };
    }
}