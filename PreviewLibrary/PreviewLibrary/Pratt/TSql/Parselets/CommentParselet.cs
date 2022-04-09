using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class CommentParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var tokenStr = parser.Scanner.GetSpanString(token);
			return new CommentSqlCodeExpr
			{
				Content = tokenStr
			};
		}
	}
}
