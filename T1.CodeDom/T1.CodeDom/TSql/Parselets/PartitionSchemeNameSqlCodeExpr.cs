using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class PartitionSchemeNameSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            Name.WriteToStream(stream);
            stream.Write("(");
            Column.WriteToStream(stream);
            stream.Write(")");
        }

        public SqlCodeExpr Name { get; set; }
        public SqlCodeExpr Column { get; set; }
    }
}