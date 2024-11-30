namespace SqlSharpLit.Common.ParserLit;

public interface ISelectColumnExpression
{
    SelectItemType ItemType { get; }
    string Alias { get; set; }
    string ToSql();
}