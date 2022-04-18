
using System.Collections.Generic;
using System.Linq;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class SetSqlCodeExpr : SqlCodeExpr
	{
		public List<string> Options { get; set; }
		public string Toggle { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("SET ");
			var optionsStr = string.Join(", ", Options.Select(x => x.ToUpper()));
			stream.Write(optionsStr);
			stream.Write($" {Toggle.ToUpper()}");
		}
	}
}