using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class IsNullParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            parser.Scanner.Consume(SqlToken.LParen);
            var checkExpression = parser.ParseExpIgnoreComment();
            parser.Scanner.Consume(SqlToken.Comma);
            var replacementValue = parser.ParseExpIgnoreComment();
            parser.Scanner.Consume(SqlToken.RParen);
            return new IsNullSqlCodeExpr
            {
                CheckExpr = checkExpression,
                ReplacementValue = replacementValue
            };
        }
    }

    public class ForXmlParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            parser.ConsumeToken(SqlToken.XML);

            var optionList = new List<XmlOptionSqlCodeExpr>();
            do
            {
                var xmlOption = parser.ConsumeTokenAny(SqlToken.AUTO, SqlToken.PATH, SqlToken.EXPLICIT, SqlToken.RAW,
                    SqlToken.ROOT);
                SqlCodeExpr rightExpr = null;
                if (xmlOption.Type == SqlToken.PATH.ToString())
                {
                    rightExpr = parser.ParseExpIgnoreComment();
                }

                if (xmlOption.Type == SqlToken.ROOT.ToString())
                {
                    rightExpr = parser.ParseExpIgnoreComment();
                }

                optionList.Add(new XmlOptionSqlCodeExpr
                {
                    Option = xmlOption.GetTokenType(),
                    RightExpr = rightExpr
                });
            } while (parser.MatchToken(SqlToken.Comma));

            return new ForXmlSqlCodeExpr
            {
                OptionList = optionList
            };
        }
    }

    public class XmlOptionSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write($"{Option.ToString().ToUpper()}");
            if (RightExpr != null)
            {
                stream.Write(" ");
                RightExpr.WriteToStream(stream);
            }
        }

        public SqlToken Option { get; set; }
        public SqlCodeExpr RightExpr { get; set; }
    }

    public class ForXmlSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write($"FOR XML ");
            OptionList.WriteToStreamWithComma(stream);
        }

        public List<XmlOptionSqlCodeExpr> OptionList { get; set; }
    }
}