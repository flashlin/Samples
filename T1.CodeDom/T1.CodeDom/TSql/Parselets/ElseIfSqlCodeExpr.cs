using System.Collections.Generic;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class ElseIfSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr ConditionExpr { get; set; }
		public List<SqlCodeExpr> Body { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("ELSE IF ");
			ConditionExpr.WriteToStream(stream);
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