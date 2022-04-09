using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using PreviewLibrary.Pratt.TSql.Parselets;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql
{
	public class TSqlParser : PrattParser
	{
		public TSqlParser(IScanner scanner) : base(scanner)
		{
			Register(SqlToken.Select, new SelectParselet());
			Register(SqlToken.Number, new NumberParselet());
			Register(SqlToken.SqlIdentifier, new ObjectIdParselet());
			Register(SqlToken.Identifier, new ObjectIdParselet());
			Register(SqlToken.MultiComment, new CommentParselet());
			Register(SqlToken.Go, new GoParselet());
			Prefix(SqlToken.PLUS, Precedence.PREFIX);
		}

		public SqlCodeExpr ParseExpression()
		{
			return (SqlCodeExpr)ParseExp(0);
		}

		protected void Register(SqlToken tokenType, IPrefixParselet parselet)
		{
			Register(tokenType.ToString(), parselet);
		}

		protected void Prefix(SqlToken tokenType, Precedence precedence)
		{
			Register(tokenType.ToString(), new SqlPrefixOperatorParselet(precedence));
		}

		protected override IPrefixParselet CodeSpecPrefix(TextSpan token)
		{
			try
			{
				return base.CodeSpecPrefix(token);
			}
			catch (KeyNotFoundException)
			{
				var tokenStr = _scanner.GetSpanString(token);

				var helpMessage = _scanner.GetHelpMessage(token);
				throw new ParseException($"Not found SqlType.{token.Type} '{tokenStr}' in PrefixParselets map.\r\n{helpMessage}");
			}
		}
	}
}
