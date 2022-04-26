using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
    public class DataTypeSqlCodeExpr : SqlCodeExpr
    {
        public SqlCodeExpr DataType { get; set; }
        public SqlCodeExpr SizeExpr { get; set; }
        public bool IsIdentity { get; set; }
        public bool IsReadOnly { get; set; }
        public List<SqlCodeExpr> ExtraList { get; set; }

        public override void WriteToStream(IndentStream stream)
        {
            DataType.WriteToStream(stream);

            if (IsIdentity)
            {
                stream.Write(" IDENTITY");
            }

            if (IsReadOnly)
            {
                stream.Write(" READONLY");
            }

            if (SizeExpr != null)
            {
                SizeExpr.WriteToStream(stream);
            }

            if (ExtraList != null && ExtraList.Count > 0)
            {
                stream.Write(" ");
                ExtraList.WriteToStream(stream);
            }
        }
    }
}