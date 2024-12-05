namespace T1.SqlSharp.Expressions;

public interface ISelectColumnExpression
{
    SelectItemType ItemType { get; }
    string Alias { get; set; }
    string ToSql();
}