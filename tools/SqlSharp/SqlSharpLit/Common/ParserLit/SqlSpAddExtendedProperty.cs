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
    public string ToSql()
    {
        return $"EXEC SP_AddExtendedProperty @name={Name}, @value={Value}, @level0type=Level0Type, @level0name=Level0Name, @level1type={Level1Type}, @level1name={Level1Name}, @level2type={Level2Type}, @level2name={Level2Name}";
    }
}