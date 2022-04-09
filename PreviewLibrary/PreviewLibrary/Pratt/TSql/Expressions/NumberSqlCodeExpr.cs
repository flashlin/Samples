using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql.Expressions
{
	public class NumberSqlCodeExpr : SqlCodeExpr
	{
		public string Value { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(Value);
		}
	}
}