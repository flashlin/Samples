namespace SqlSharpLit.Common.ParserLit;

public class SelectFrom : ISelectFromExpression
{
    public string FromTableName { get; set; } = string.Empty;
}