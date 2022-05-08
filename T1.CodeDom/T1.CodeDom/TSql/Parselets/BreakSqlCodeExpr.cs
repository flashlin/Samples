using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class BreakSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("BREAK");
            if (IsSemicolon)
            {
                stream.Write(" ;");
            }
        }

        public bool IsSemicolon { get; set; }
    }
}