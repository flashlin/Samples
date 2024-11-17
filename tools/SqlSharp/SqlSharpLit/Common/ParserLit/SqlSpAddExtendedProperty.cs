namespace SqlSharpLit.Common.ParserLit;

public class SqlSpAddExtendedProperty : ISqlExpression
{
    public string Name { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;
    public string Level0Type { get; set; } = string.Empty;
    public string Level0Name { get; set; } = string.Empty;
    public string Level1Type { get; set; } = string.Empty;
    public string Level1Name { get; set; } = string.Empty;
    public string Level2Type { get; set; } = string.Empty;
    public string Level2Name { get; set; } = string.Empty;
}