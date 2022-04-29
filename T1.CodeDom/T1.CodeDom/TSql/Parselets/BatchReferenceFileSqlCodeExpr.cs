using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class BatchReferenceFileSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write($":r {File}");
        }

        public string File { get; set; }
    }
}