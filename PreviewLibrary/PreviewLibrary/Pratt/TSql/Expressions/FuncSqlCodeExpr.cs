using PreviewLibrary.Pratt.Core.Expressions;
using System.Collections.Generic;
using System.Linq;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class FuncSqlCodeExpr : SqlCodeExpr
	{
		public string Name { get; set; }
		public List<SqlCodeExpr> Parameters { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write($"{Name}(");
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