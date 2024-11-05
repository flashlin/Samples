namespace SqlSharpLit.Common.ParserLit;

public class TextSpan
{
    public string Word { get; set; } = string.Empty;
    public int Offset { get; set; }
    public int Length { get; set; }
}