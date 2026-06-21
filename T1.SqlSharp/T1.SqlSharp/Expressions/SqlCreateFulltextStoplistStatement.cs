namespace T1.SqlSharp.Expressions;

public class SqlCreateFulltextStoplistStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateFulltextStoplistStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateFulltextStoplistStatement(this);
    }

    public string Name { get; set; } = string.Empty;
    public string Source { get; set; } = string.Empty;

    public string ToSql()
    {
        var source = string.IsNullOrEmpty(Source) ? string.Empty : $" FROM {Source}";
        return $"CREATE FULLTEXT STOPLIST {Name}{source}";
    }
}
