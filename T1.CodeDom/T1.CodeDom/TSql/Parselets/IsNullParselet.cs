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
			var xmlOption = parser.ConsumeTokenAny(SqlToken.AUTO, SqlToken.PATH, SqlToken.EXPLICIT, SqlToken.RAW);

			SqlCodeExpr rightExpr = null;
			if (xmlOption.Type == SqlToken.PATH.ToString())
			{
				rightExpr = parser.ParseExpIgnoreComment();
			}

			return new ForXmlSqlCodeExpr
			{
				Option = xmlOption.GetTokenType(),
				RightExpr = rightExpr
			};
		}
	}

	public class ForXmlSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write($"FOR XML {Option.ToString().ToUpper()}");
			if (RightExpr != null)
			{
				RightExpr.WriteToStream(stream);
			}
		}

		public SqlToken Option { get; set; }
		public SqlCodeExpr RightExpr { get; set; }
	}
}
