using System;
using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;
using T1.Standard.Net.SoapProtocols.WsdlXmlDeclrs;

namespace T1.CodeDom.TSql.Parselets
{
    public class NumberParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            var tokenStr = parser.Scanner.GetSpanString(token);

            var startIndex = parser.Scanner.GetOffset();
            if (tokenStr.Length == 4 && parser.MatchToken(SqlToken.Minus))
            {
                var month = ParseNumber(2, parser);
                var minus = parser.ConsumeToken();
                var day = ParseNumber(2, parser);
                if (!string.IsNullOrEmpty(month) && minus.Type == SqlToken.Minus.ToString() &&
                    !string.IsNullOrEmpty(day))
                {
                    return new DateSqlCodeExpr
                    {
                        Value = $"{tokenStr}-{month}-{day}"
                    };
                }
                parser.Scanner.SetOffset(startIndex);
            }

            return new NumberSqlCodeExpr
            {
                Value = tokenStr
            };
        }

        private string ParseNumber(int n, IParser parser)
        {
            var tokenSpan = parser.PeekToken();
            if (tokenSpan.Type != SqlToken.Number.ToString())
            {
                return string.Empty;
            }

            if (tokenSpan.Length != n)
            {
                return string.Empty;
            }

            parser.ConsumeToken();
            return parser.Scanner.GetSpanString(tokenSpan);
        }
    }

    public class DateSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write(Value);
        }

        public string Value { get; set; }
    }

    public class ValuesParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            var itemList = new List<SqlCodeExpr>();
            parser.ConsumeToken(SqlToken.LParen);
            do
            {
                var item = parser.ParseExpIgnoreComment();
                itemList.Add(item);
            } while (parser.MatchToken(SqlToken.Comma));

            parser.ConsumeToken(SqlToken.RParen);

            return new ValuesSqlCodeExpr
            {
                ValueList = itemList
            };
        }
    }
}