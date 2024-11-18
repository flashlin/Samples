namespace SqlSharpLit.Common.ParserLit;

public class SelectFrom : ISelectFromExpression
{
    public string FromTableName { get; set; } = string.Empty;
    public string ToSql()
    {
        return FromTableName;
    }
}