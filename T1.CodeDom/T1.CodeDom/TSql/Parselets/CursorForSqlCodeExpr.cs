using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class CursorForSqlCodeExpr : SqlCodeExpr 
	{
		public SqlCodeExpr SelectExpr { get; set; }
		public List<SqlToken> Options { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CURSOR");

			if (Options != null && Options.Count > 0)
			{
				stream.Write(" ");
				var code = string.Join(" ", Options);
				stream.Write(code);
			}
			
			stream.Write(" FOR ");
			SelectExpr.WriteToStream(stream);
		}
	}
}