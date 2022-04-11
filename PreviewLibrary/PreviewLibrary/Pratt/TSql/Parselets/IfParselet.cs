using System.Collections.Generic;
using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
    public class IfParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            var conditionExpr = parser.ParseExp();
            parser.Scanner.Consume(SqlToken.Begin);
            var bodyList = new List<SqlCodeExpr>();
            do
            {
                var body = parser.ParseExp();
                bodyList.Add(body as SqlCodeExpr);
            } while (!parser.Scanner.TryConsumeTokenType(SqlToken.End, out _));

            return new IfSqlCodeExpr
            {
                Condition = conditionExpr as SqlCodeExpr,
                Body = bodyList
            };
        }
    }
}