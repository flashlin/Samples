namespace SqlSharpLit.Common.ParserLit;

public class SqlIdentity
{
    public static SqlIdentity Default => new();
    public int Seed { get; set; }
    public int Increment { get; set; }
}