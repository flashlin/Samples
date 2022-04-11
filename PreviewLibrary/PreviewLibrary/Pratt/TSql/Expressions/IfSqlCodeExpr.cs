using System.Collections.Generic;
using PreviewLibrary.Pratt.Core.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class IfSqlCodeExpr : SqlCodeExpr
	{
		public List<SqlCodeExpr> Body { get; set; }
		public SqlCodeExpr Condition { get; set; }
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
			stream.WriteLine("END");
		}
	}
}