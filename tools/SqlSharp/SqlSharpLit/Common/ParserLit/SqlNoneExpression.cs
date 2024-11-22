namespace SqlSharpLit.Common.ParserLit;

public class SqlNoneExpression : ISqlExpression
{
    public static SqlNoneExpression Default { get; } = new();
    public SqlType SqlType => SqlType.None;
    public string ToSql()
    {
        return string.Empty;
    }
}

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