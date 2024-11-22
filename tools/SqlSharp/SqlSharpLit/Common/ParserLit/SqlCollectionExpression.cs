namespace SqlSharpLit.Common.ParserLit;

public class SqlCollectionExpression : ISqlExpression
{
    public SqlType SqlType => SqlType.Collection;
    public List<ISqlExpression> Items { get; set; } = [];

    public List<T> ToList<T>()
    {
        return Items.Cast<T>().ToList();
    }
    public string ToSql()
    {
        return string.Empty;
    }
}