using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class ConstantSqlCodeExpr : SqlCodeExpr
	{
		public string Value { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(Value);
		}
	}
}