using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class BeginSqlCodeExpr : SqlCodeExpr 
	{
		public List<SqlCodeExpr> Items { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.WriteLine("BEGIN");
			Items.WriteToStream(stream);
			stream.WriteLine();
			stream.Write("END");
		}
	}
}