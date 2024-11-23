namespace SqlSharpLit.Common.ParserLit;

public class TextSpan
{
    public static TextSpan None { get; } = new();
    public string Word { get; set; } = string.Empty;
    public int Offset { get; set; } = -1;
    public int Length { get; set; }
}