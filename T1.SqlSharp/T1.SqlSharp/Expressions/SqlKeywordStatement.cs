namespace T1.SqlSharp.Expressions;

public class SqlKeywordStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.KeywordStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_KeywordStatement(this);
    }

    public string Keyword { get; set; } = string.Empty;

    public string ToSql()
    {
        return Keyword;
    }
}
