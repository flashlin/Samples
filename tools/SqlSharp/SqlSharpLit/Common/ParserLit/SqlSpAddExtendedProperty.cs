namespace SqlSharpLit.Common.ParserLit;

public class SqlSpAddExtendedProperty : ISqlExpression
{
    public SqlType SqlType => SqlType.AddExtendedProperty;
    /// <summary>
    /// 固定為 'MS_Description' 來表示描述
    /// </summary>
    public string Name { get; set; } = string.Empty;
    /// <summary>
    /// 欄位的描述內容 
    /// </summary>
    public string Value { get; set; } = string.Empty;
    /// <summary>
    /// 第0層的類型 (通常為 Schema)
    /// </summary>
    public string Level0Type { get; set; } = string.Empty;
    /// <summary>
    /// Schema 名稱 (通常為: dbo)
    /// </summary>
    public string Level0Name { get; set; } = string.Empty;
    /// <summary>
    /// 第1層的類型 (通常為: TABLE)
    /// </summary>
    public string Level1Type { get; set; } = string.Empty;
    /// <summary>
    /// 表格名稱: (Customer, Product, ...)
    /// </summary>
    public string Level1Name { get; set; } = string.Empty;
    /// <summary>
    /// 第2層的類型 (通常固定為 COLUMN)
    /// </summary>
    public string Level2Type { get; set; } = string.Empty;
    /// <summary>
    /// 欄位名稱 (Address, Phone, ...)
    /// </summary>
    public string Level2Name { get; set; } = string.Empty;
    public string ToSql()
    {
        return $"EXEC SP_AddExtendedProperty @name={Name}, @value={Value}, @level0type=Level0Type, @level0name=Level0Name, @level1type={Level1Type}, @level1name={Level1Name}, @level2type={Level2Type}, @level2name={Level2Name}";
    }
}