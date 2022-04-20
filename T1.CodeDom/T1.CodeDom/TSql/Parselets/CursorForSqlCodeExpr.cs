using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class CursorForSqlCodeExpr : SqlCodeExpr 
	{
		public SqlCodeExpr SelectExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CURSOR FOR ");
			SelectExpr.WriteToStream(stream);
		}
	}
}