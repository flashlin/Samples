using System.Collections.Generic;
using System.Linq;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class FuncSqlCodeExpr : SqlCodeExpr
	{
		public SqlCodeExpr Name { get; set; }
		public List<SqlCodeExpr> Parameters { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			Name.WriteToStream(stream);
			stream.Write("(");
			if (Parameters.Count > 0)
			{
				stream.Write(" ");
				foreach (var parameter in Parameters.Select((val, idx) => new { val, idx }))
				{
					if (parameter.idx != 0)
					{
						stream.Write(", ");
					}
					parameter.val.WriteToStream(stream);
				}
				stream.Write(" ");
			}
			stream.Write(")");
		}
	}
}