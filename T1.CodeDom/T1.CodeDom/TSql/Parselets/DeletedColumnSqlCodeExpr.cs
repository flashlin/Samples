using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class DeletedColumnSqlCodeExpr : SqlCodeExpr
    {
        public SqlCodeExpr ColumnExpr { get; set; }

        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("DELETED.");
            ColumnExpr.WriteToStream(stream);
        }
    }
}