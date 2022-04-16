using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class RankSqlCodeExpr : SqlCodeExpr 
	{
		public List<SqlCodeExpr> OrderByClause { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("RANK() OVER(");
			stream.WriteLine();
			stream.Indent++;

			stream.Write("ORDER BY ");
			OrderByClause.WriteToStreamWithComma(stream);

			stream.Indent--;
			stream.Write(")");
		}
	}
}