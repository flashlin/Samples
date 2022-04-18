using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class CreateTableSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public List<SqlCodeExpr> ColumnsList { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("CREATE TABLE ");
			Name.WriteToStream(stream);
			stream.WriteLine("(");
			ColumnsList.WriteToStreamWithCommaLine(stream);
			stream.WriteLine();
			stream.Write(")");
		}
	}
}