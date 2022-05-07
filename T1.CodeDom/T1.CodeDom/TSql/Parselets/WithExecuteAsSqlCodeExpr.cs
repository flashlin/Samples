using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class WithExecuteAsSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write($"WITH EXECUTE AS {UserExpr}");
        }

        public string UserExpr { get; set; }
    }
}