using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class ForLoginSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("FOR LOGIN ");
            LoginName.WriteToStream(stream);
        }

        public SqlCodeExpr LoginName { get; set; }
    }
}