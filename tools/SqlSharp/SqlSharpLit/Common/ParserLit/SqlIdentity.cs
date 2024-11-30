using SqlSharpLit.Common.ParserLit.Expressions;
using T1.SqlSharp.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SqlIdentity : ISqlExpression 
{
    public SqlType SqlType { get; } = SqlType.Identity;
    public static SqlIdentity Default => new();
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