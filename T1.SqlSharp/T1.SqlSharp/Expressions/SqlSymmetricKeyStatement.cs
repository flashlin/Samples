namespace T1.SqlSharp.Expressions;

public class SqlSymmetricKeyStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.SymmetricKeyStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_SymmetricKeyStatement(this);
    }

    public bool IsOpen { get; set; }
    public bool AllKeys { get; set; }
    public string KeyName { get; set; } = string.Empty;
    public string DecryptionBy { get; set; } = string.Empty;

    public string ToSql()
    {
        var verb = IsOpen ? "OPEN" : "CLOSE";
        if (AllKeys)
        {
            return $"{verb} ALL SYMMETRIC KEYS";
        }

        var decryption = string.IsNullOrEmpty(DecryptionBy) ? string.Empty : $" DECRYPTION BY {DecryptionBy}";
        return $"{verb} SYMMETRIC KEY {KeyName}{decryption}";
    }
}
