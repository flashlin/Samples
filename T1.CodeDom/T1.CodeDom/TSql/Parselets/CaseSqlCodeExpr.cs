using System.Collections.Generic;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class CaseSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr InputExpr { get; set; }
		public List<SqlCodeExpr> WhenList { get; set; }
		public SqlCodeExpr ElseExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CASE");

			if( InputExpr != null )
			{
				stream.Write(" ");
				InputExpr.WriteToStream(stream);
			}

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