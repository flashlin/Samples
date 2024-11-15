namespace SqlSharpLit.Common.ParserLit;

public class TextSpan
{
    public string Word { get; set; } = string.Empty;
    public int Offset { get; set; } = -1;
    public int Length { get; set; }
}