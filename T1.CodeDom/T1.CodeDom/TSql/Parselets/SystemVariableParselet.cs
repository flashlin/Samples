using PreviewLibrary.Pratt.TSql.Expressions;
using T1.CodeDom.Core;

namespace T1.CodeDom.TSql.Parselets
{
	public class SystemVariableParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var name = parser.Scanner.GetSpanString(token);
			return new SystemVariableSqlCodeExpr
			{
				Name = name,
			};
		}
	}
}
