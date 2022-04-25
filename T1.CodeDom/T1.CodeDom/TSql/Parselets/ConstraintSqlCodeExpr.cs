using System.Collections.Generic;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class ConstraintSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("CONSTRAINT ");
            ConstraintName.WriteToStream(stream);
            stream.Write($" {KeyType}");
            stream.Write($" {ClusterType}");
            stream.Write("(");
            ColumnList.WriteToStreamWithComma(stream);
            stream.Write(")");
        }

        public SqlCodeExpr ConstraintName { get; set; }
        public string KeyType { get; set; }
        public string ClusterType { get; set; }
        public List<OrderItemSqlCodeExpr> ColumnList { get; set; }
    }
}