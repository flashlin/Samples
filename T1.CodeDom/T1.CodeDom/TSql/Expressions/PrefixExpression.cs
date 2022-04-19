using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
	public class PrefixExpression : SqlCodeExpr
	{
		public string Token { get; set; }
		public SqlCodeExpr Right { get; set; }

		public override void WriteToStream(IndentStream stream)
		{
			stream.Write(Token);
			Right.WriteToStream(stream);
		}
	}
}
