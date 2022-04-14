using System.Collections.Generic;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class DataTableTypeSqlCodeExpr : SqlCodeExpr
	{
		public List<SqlCodeExpr> Columns { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.WriteLine("TABLE");
			stream.WriteLine("(");
			Columns.WriteToStreamWithComma(stream);
			stream.WriteLine();
			stream.Write(")");
		}
	}
}