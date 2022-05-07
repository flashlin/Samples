using System.Collections.Generic;
using System.Linq;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
    public class DeclareParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            var variableList = new List<SqlCodeExpr>();
            do
            {
                var varName = parser.ParseExpIgnoreComment(int.MaxValue);

                parser.Scanner.Match(SqlToken.As);

                SqlCodeExpr dataTypeExpr = ParseCursorFor(parser);
                if (dataTypeExpr == null)
                {
                    dataTypeExpr = parser.ConsumeDataType();
                }

                SqlCodeExpr variableDataType = new DeclareSqlCodeExpr
                {
                    Name = varName as SqlCodeExpr,
                    DataType = dataTypeExpr,
                };

                if (parser.Scanner.Match(SqlToken.Equal))
                {
                    var valueExpr = parser.ParseExpIgnoreComment();
                    variableDataType = new AssignSqlCodeExpr
                    {
                        Left = variableDataType,
                        Oper = "=",
                        Right = valueExpr
                    };
                }

                variableList.Add(variableDataType);
            } while (parser.Scanner.Match(SqlToken.Comma));

            return new ExprListSqlCodeExpr
            {
                IsComma = false,
                Items = variableList
            };
        }

        static CursorForSqlCodeExpr ParseCursorFor(IParser parser)
        {
            var startIndex = parser.Scanner.GetOffset();
            if (!parser.MatchToken(SqlToken.Cursor))
            {
                return null;
            }

            var results = EachTryConsumeTokenAny(parser,
                new[] {SqlToken.LOCAL, SqlToken.GLOBAL},
                new[] {SqlToken.FORWARD_ONLY, SqlToken.SCROLL},
                new[] {SqlToken.STATIC, SqlToken.KEYSET, SqlToken.DYNAMIC, SqlToken.FAST_FORWARD},
                new[] {SqlToken.READ_ONLY, SqlToken.SCROLL_LOCKS, SqlToken.OPTIMISTIC},
                new[] {SqlToken.TYPE_WARNING});

            if (!parser.MatchToken(SqlToken.FOR))
            {
                parser.Scanner.SetOffset(startIndex);
                return null;
            }
            
            var expr = parser.ParseExpIgnoreComment();

            return new CursorForSqlCodeExpr
            {
                Options = results.Select(x => x.GetTokenType()).ToList(),
                SelectExpr = expr
            };
        }

        private static List<TextSpan> EachTryConsumeTokenAny(IParser parser, params SqlToken[][] tokenTypesList)
        {
            var results = new List<TextSpan>();
            foreach (var tokenTypes in tokenTypesList)
            {
                if (parser.TryConsumeTokenAny(out var span, tokenTypes))
                {
                    results.Add(span);
                }
            }

            return results;
        }
    }
}