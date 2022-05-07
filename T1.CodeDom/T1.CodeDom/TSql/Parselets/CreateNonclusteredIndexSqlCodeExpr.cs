using System.Collections.Generic;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class CreateNonclusteredIndexSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("CREATE");

            if (IsUnique)
            {
                stream.Write(" UNIQUE");
            }

            stream.Write(" NONCLUSTERED INDEX ");
            IndexName.WriteToStream(stream);
            stream.Write(" ON ");
            TableName.WriteToStream(stream);
            stream.Write("(");
            ColumnList.WriteToStreamWithComma(stream);
            stream.Write(")");

            if (WhereExpr != null)
            {
                stream.Write(" WHERE (");
                WhereExpr.WriteToStream(stream);
                stream.Write(")");
            }

            if (WithExpr != null)
            {
                stream.Write(" ");
                WithExpr.WriteToStream(stream);
            }

            if (OnPrimary != null)
            {
                stream.Write(" ");
                OnPrimary.WriteToStream(stream);
            }

            if (IsSemicolon)
            {
                stream.Write(" ;");
            }
        }

        public SqlCodeExpr IndexName { get; set; }
        public SqlCodeExpr TableName { get; set; }
        public List<OrderItemSqlCodeExpr> ColumnList { get; set; }
        public bool IsSemicolon { get; set; }
        public SqlCodeExpr WhereExpr { get; set; }
        public SqlCodeExpr WithExpr { get; set; }
        public OnSqlCodeExpr OnPrimary { get; set; }
        public bool IsUnique { get; set; }
    }
}