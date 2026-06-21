namespace T1.SqlSharp.Expressions;

public class SqlSignatureStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.SignatureStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_SignatureStatement(this);
    }

    public bool IsAdd { get; set; }
    public string Target { get; set; } = string.Empty;
    public string By { get; set; } = string.Empty;

    public string ToSql()
    {
        var verb = IsAdd ? "ADD" : "DROP";
        return $"{verb} SIGNATURE TO {Target} BY {By}";
    }
}
