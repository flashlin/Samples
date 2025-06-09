namespace T1.SqlSharp.Expressions;

public class SqlIdentity : ISqlExpression 
{
    public static SqlIdentity Default => new();
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_Identity(this);
    }

    public SqlType SqlType { get; } = SqlType.Identity;
    public long Seed { get; set; }
    public int Increment { get; set; }

    public string ToSql()
    {
        if (Increment == 0)
        {
            return string.Empty;
        }
        if(Seed == 1 && Increment == 1)
        {
            return "IDENTITY";
        }
        return $"IDENTITY({Seed}, {Increment})";
    }
}