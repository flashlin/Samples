namespace T1.SqlSharp.Expressions;

public class SqlCreateXmlSchemaCollectionStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateXmlSchemaCollectionStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateXmlSchemaCollectionStatement(this);
    }

    public string Name { get; set; } = string.Empty;
    public string Schema { get; set; } = string.Empty;

    public string ToSql()
    {
        return $"CREATE XML SCHEMA COLLECTION {Name} AS {Schema}";
    }
}
