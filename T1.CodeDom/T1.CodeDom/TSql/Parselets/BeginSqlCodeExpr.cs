using System.Collections.Generic;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
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