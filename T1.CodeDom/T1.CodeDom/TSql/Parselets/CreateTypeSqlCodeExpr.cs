using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class CreateTypeSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("CREATE TYPE ");
            Name.WriteToStream(stream);
            stream.Write(" AS ");
            TypeExpr.WriteToStream(stream);
        }

        public SqlCodeExpr Name { get; set; }
        public TableDataTypeSqlCodeExpr TypeExpr { get; set; }
    }
}