using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class IfSqlCodeExpr : SqlCodeExpr
	{
		public List<SqlCodeExpr> Body { get; set; }
		public SqlCodeExpr Condition { get; set; }
		public List<SqlCodeExpr> ElseIfList { get; set; }
		public List<SqlCodeExpr> ElseExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("IF ");
			Condition.WriteToStream(stream);
			stream.WriteLine();
			stream.WriteLine("BEGIN");
			stream.Indent++;
			foreach (SqlCodeExpr expr in Body)
			{
				expr.WriteToStream(stream);
				stream.WriteLine();
			}
			stream.Indent--;
			stream.Write("END");

			if( ElseIfList != null && ElseIfList.Count > 0)
			{
				stream.WriteLine();
				ElseIfList.WriteToStream(stream);
			}

			if( ElseExpr != null && ElseExpr.Count > 0)
			{
				stream.WriteLine();
				stream.WriteLine("ELSE BEGIN");
				stream.Indent++;
				ElseExpr.WriteToStream(stream);
				stream.Indent--;
				stream.WriteLine();
				stream.Write("END");
			}
		}
	}
}