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
			Register(SqlToken.Variable, new VariableParselet());
			Register(SqlToken.NString, new NStringParselet());
			Register(SqlToken.MultiComment, new CommentParselet());
			Register(SqlToken.Go, new GoParselet());
			Register(SqlToken.Set, new SetParselet());
			Register(SqlToken.Semicolon, new SemicolonParselet());
			Register(SqlToken.ScriptSetVar, new ScriptSetvarParselet());
			Register(SqlToken.ScriptOn, new ScriptOnParselet());
			Register(SqlToken.DoubleQuoteString, new DoubleQuoteStringParselet());
			Register(SqlToken.QuoteString, new QuoteStringParselet());
			Register(SqlToken.LParen, new GroupParselet());
			Register(SqlToken.Not, new NotParselet());
			Register(SqlToken.Exists, new ExistsParselet());
			Register(SqlToken.Exec, new ExecParselet());

			Register(SqlToken.Not, new NotInfixParselet());
			Register(SqlToken.If, new IfParselet());
			
			Prefix(SqlToken.PLUS, Precedence.PREFIX);
			InfixLeft(SqlToken.Equal, Precedence.SUM);
			InfixLeft(SqlToken.And, Precedence.CONCAT);
			InfixLeft(SqlToken.Or, Precedence.CONCAT);
		}

		public SqlCodeExpr ParseExpression()
		{
			return (SqlCodeExpr)ParseExp(0);
		}

		protected void Register(SqlToken tokenType, IPrefixParselet parselet)
		{
			Register(tokenType.ToString(), parselet);
		}
		
		protected void Register(SqlToken tokenType, IInfixParselet parselet)
		{
			Register(tokenType.ToString(), parselet);
		}

		protected void Prefix(SqlToken tokenType, Precedence precedence)
		{
			Register(tokenType.ToString(), new SqlPrefixOperatorParselet(precedence));
		}

		public void InfixLeft(SqlToken tokenType, Precedence precedence)
		{
			Register(tokenType.ToString(), new BinaryOperatorParselet(precedence, false));
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
