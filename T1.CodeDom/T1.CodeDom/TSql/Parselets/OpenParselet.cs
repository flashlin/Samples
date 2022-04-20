using T1.CodeDom.Core;

namespace T1.CodeDom.TSql.Parselets
{
	public class OpenParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var cursorName = parser.ConsumeTableName();

			return new OpenSqlCodeExpr
			{
				CursorName = cursorName
			};
		}
	}
}