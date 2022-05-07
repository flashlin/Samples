using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class CreateSynonymSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("CREATE SYNONYM ");
            Name.WriteToStream(stream);
            stream.Write(" FOR ");
            ObjectId.WriteToStream(stream);
        }

        public SqlCodeExpr Name { get; set; }
        public SqlCodeExpr ObjectId { get; set; }
    }
}