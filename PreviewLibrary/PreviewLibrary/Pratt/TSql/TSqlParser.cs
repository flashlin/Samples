using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.TSql.Expressions;
using PreviewLibrary.Pratt.TSql.Parselets;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql
{
	public class TSqlParser : PrattParser
	{
		public TSqlParser(IScanner scanner) : base(scanner)
		{
			Prefix(SqlToken.PLUS, Precedence.PREFIX);
		}

		public SqlCodeExpr ParseExpression()
		{
			return (SqlCodeExpr)ParseExp(0);
		}

		protected void Prefix(SqlToken tokenType, Precedence precedence)
		{
			Register(tokenType.ToString(), new SqlPrefixOperatorParselet(precedence));
		}

		protected override PrefixParselet CodeSpecPrefix(TextSpan token)
		{
			try
			{
				return base.CodeSpecPrefix(token);
			}
			catch (KeyNotFoundException)
			{
				var tokenStr = _scanner.GetSpanString(token);
				throw new ParseException($"Not found SqlType.{token.Type} '{tokenStr}' in PrefixParselets map.");
			}
		}
	}
}
