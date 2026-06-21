namespace T1.SqlSharp.Expressions;

public class SqlXmlNamespacesStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.XmlNamespacesStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_XmlNamespacesStatement(this);
    }

    public List<string> Namespaces { get; set; } = [];
    public ISqlExpression? Statement { get; set; }

    public string ToSql()
    {
        var body = Statement?.ToSql() ?? string.Empty;
        return $"WITH XMLNAMESPACES ({string.Join(", ", Namespaces)}) {body}";
    }
}
