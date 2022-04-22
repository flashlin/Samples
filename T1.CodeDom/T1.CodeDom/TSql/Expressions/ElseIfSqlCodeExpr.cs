using System.Collections.Generic;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class ElseIfSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr ConditionExpr { get; set; }
		//public List<SqlCodeExpr> Body { get; set; }
		public SqlCodeExpr Body { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("ELSE ");

			if (ConditionExpr != null)
			{
				stream.Write("IF ");
				ConditionExpr.WriteToStream(stream);
				stream.WriteLine();
			}

			Body.WriteToStream(stream);
		}
	}
}