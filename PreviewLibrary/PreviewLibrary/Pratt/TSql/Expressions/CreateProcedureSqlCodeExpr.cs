using PreviewLibrary.Pratt.Core.Expressions;
using System.Collections.Generic;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class CreateProcedureSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public List<ArgumentSqlCodeExpr> Arguments { get; set; }
		public List<SqlCodeExpr> Body { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CREATE PROCEDURE ");
			Name.WriteToStream(stream);
			stream.WriteLine();
			Arguments.WriteToStreamWithComma(stream);
			stream.WriteLine();
			stream.WriteLine("BEGIN");
			stream.Indent++;
			Body.WriteToStream(stream);
			stream.Indent--;
			stream.WriteLine();
			stream.Write("END");
		}
	}
}