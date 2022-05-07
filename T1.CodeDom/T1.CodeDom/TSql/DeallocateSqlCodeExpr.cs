using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql
{
    public class DeallocateSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("DEALLOCATE ");
            Name.WriteToStream(stream);
        }

        public SqlCodeExpr Name { get; set; }
    }
}