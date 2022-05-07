using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class CreateViewSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("CREATE VIEW ");
            Name.WriteToStream(stream);
            stream.WriteLine();
            stream.WriteLine("AS ");
            Expr.WriteToStream(stream);
        }

        public SqlCodeExpr Name { get; set; }
        public SqlCodeExpr Expr { get; set; }
    }
}