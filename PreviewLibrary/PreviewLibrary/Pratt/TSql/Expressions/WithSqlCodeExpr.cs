using PreviewLibrary.Pratt.Core.Expressions;
using System.Collections.Generic;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class WithSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Table { get; set; }
		public List<SqlCodeExpr> Columns { get; set; }
		public SqlCodeExpr InnerExpr { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("WITH ");
			Table.WriteToStream(stream);

			stream.Write("(");
			Columns.WriteToStreamWithComma(stream);
			stream.WriteLine(")");

			stream.WriteLine("AS (");
			stream.Indent++;
			InnerExpr.WriteToStream(stream);
			stream.Indent--;
			stream.WriteLine();
			stream.Write(")");
		}
	}
}