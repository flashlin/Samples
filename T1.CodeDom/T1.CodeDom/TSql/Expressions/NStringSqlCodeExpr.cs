using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class NStringSqlCodeExpr : SqlCodeExpr
	{
		public string Value { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(Value);
		}
	}
}