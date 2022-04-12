using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using System.Linq;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class ExecSqlCodeExpr : SqlCodeExpr
	{
		public string ExecToken { get; set; }
		public SqlCodeExpr Name { get; set; }
		public List<SqlCodeExpr> Parameters { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write($"{ExecToken} ");
			Name.WriteToStream(stream);
			stream.Write(" ");
			Parameters.WriteToStreamWithComma(stream);
		}
	}
}