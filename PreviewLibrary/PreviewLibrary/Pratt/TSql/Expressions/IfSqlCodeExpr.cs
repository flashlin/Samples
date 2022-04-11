using System.Collections.Generic;
using PreviewLibrary.Pratt.Core.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
    public class IfSqlCodeExpr : SqlCodeExpr 
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("IF ");
			
        }

        public IExpression Condition { get; set; }
        public List<SqlCodeExpr> Body { get; set; }
    }
}