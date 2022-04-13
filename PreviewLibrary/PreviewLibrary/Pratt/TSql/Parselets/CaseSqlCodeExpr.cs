using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class CaseSqlCodeExpr : SqlCodeExpr
	{
		public List<SqlCodeExpr> WhenList { get; set; }
		public SqlCodeExpr ElseExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CASE ");
			foreach (var whenExpr in WhenList)
			{
				stream.WriteLine();
				whenExpr.WriteToStream(stream);
			}

			if (ElseExpr != null)
			{
				stream.WriteLine();
				stream.Write("ELSE ");
				ElseExpr.WriteToStream(stream);
			}

			stream.WriteLine();
			stream.Write("END");
		}
	}
}