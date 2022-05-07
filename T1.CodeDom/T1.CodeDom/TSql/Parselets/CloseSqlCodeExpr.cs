using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class CloseSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("CLOSE ");
            Name.WriteToStream(stream);
        }

        public SqlCodeExpr Name { get; set; }
    }
}