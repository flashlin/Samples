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

            if (ClusterExpr != null)
            {
                stream.Write(" ");
                ClusterExpr.WriteToStream(stream);
            }

            if (WithExpr != null)
            {
                stream.Write(" ");
                WithExpr.WriteToStream(stream);
            }
        }

        public SqlCodeExpr ConstraintName { get; set; }
        public string KeyType { get; set; }
        public SqlCodeExpr WithExpr { get; set; }
        public ClusteredSqlCodeExpr ClusterExpr { get; set; }
    }
}