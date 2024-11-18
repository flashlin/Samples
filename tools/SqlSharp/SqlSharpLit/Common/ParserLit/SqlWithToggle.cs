namespace SqlSharpLit.Common.ParserLit;

public class SqlWithToggle
{
    public string ToggleName { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"{ToggleName} {Value}";
    }
}