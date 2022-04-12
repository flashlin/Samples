using System.Collections.Generic;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class ExprListSqlCodeExpr : SqlCodeExpr
	{
		public List<SqlCodeExpr> Items { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Items.WriteToStreamWithComma(stream);
		}
	}
}