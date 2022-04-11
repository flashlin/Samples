using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class NotInfixParselet : IInfixParselet
    {
        public IExpression Parse(IExpression left, TextSpan token, IParser parser)
        {
            if (parser.Scanner.TryConsumeAny(out var nextToken, SqlToken.Like))
            {
                var right = parser.ParseExp();
                return new NotLikeSqlCodeExpr
                {
                    Left = left as SqlCodeExpr,
                    Right = right as SqlCodeExpr
                };
            }

            var helpMessage = parser.Scanner.GetHelpMessage(token);
            throw new ParseException(helpMessage);
        }

        public int GetPrecedence()
        {
            return (int)Precedence.CALL;
        }
    }
}