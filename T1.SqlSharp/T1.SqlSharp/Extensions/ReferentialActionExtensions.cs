using T1.SqlSharp.Expressions;

namespace T1.SqlSharp.Extensions;

public static class ReferentialActionExtensions
{
    public static string ToSql(this ReferentialAction action)
    {
        return action switch
        {
            ReferentialAction.Cascade => "CASCADE",
            ReferentialAction.SetNull => "SET NULL",
            ReferentialAction.SetDefault => "SET DEFAULT",
            _ => "NO ACTION"
        };
    }
}