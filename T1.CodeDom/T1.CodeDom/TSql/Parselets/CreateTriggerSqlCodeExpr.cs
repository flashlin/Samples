using System.Collections.Generic;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class CreateTriggerSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("CREATE ");
            TriggerExpr.WriteToStream(stream);

            if (ForTableExpr != null)
            {
                stream.Write(" FOR ");
                ForTableExpr.WriteToStream(stream);
            }

            stream.WriteLine();
            stream.WriteLine("AS ");
            Body.WriteToStream(stream);
        }

        public SqlCodeExpr TriggerExpr { get; set; }
        public List<SqlCodeExpr> Body { get; set; }
        public SqlCodeExpr ForTableExpr { get; set; }
    }
}