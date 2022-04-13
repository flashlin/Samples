using PreviewLibrary.Pratt.Core.Expressions;
using System.Collections.Generic;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class UpdateSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Table { get; set; }
		public List<SqlCodeExpr> SetColumnsList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("UPDATE ");
			Table.WriteToStream(stream);
			stream.Write(" SET ");
			SetColumnsList.WriteToStreamWithComma(stream);
		}
	}
}