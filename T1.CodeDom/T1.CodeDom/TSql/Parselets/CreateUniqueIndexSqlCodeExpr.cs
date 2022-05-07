using System.Collections.Generic;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class CreateUniqueIndexSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("CREATE UNIQUE INDEX ");
            IndexName.WriteToStream(stream);
            stream.Write(" ON ");
            TableName.WriteToStream(stream);
            stream.Write("(");
            OnColumns.WriteToStreamWithComma(stream);
            stream.Write(")");
        }

        public SqlCodeExpr IndexName { get; set; }
        public SqlCodeExpr TableName { get; set; }
        public List<SqlCodeExpr> OnColumns { get; set; }
    }
}