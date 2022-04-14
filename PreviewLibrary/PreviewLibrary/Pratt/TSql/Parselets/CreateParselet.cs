using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Linq;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class CreateParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			if (parser.Scanner.Match(SqlToken.Procedure))
			{
				return CreateProcedure(token, parser);
			}

			if(parser.Scanner.Match(SqlToken.Function))
			{
				return CreateFunction(token, parser);
			}

			var helpMessage = parser.Scanner.GetHelpMessage();
			throw new ParseException($"Parse CREATE Error, {helpMessage}");
		}

		private IExpression CreateFunction(TextSpan token, IParser parser)
		{
			var nameExpr = parser.Scanner.ConsumeObjectId();
			parser.Scanner.Consume(SqlToken.LParen);
			var arguments = parser.ConsumeArgumentList();
			parser.Scanner.Consume(SqlToken.RParen);

			parser.Scanner.Consume(SqlToken.Returns);

			parser.Scanner.TryConsumeVariable(out var returnVariableExpr);


			var returnTypeExpr = parser.ConsumeDataType();

			parser.Scanner.Consume(SqlToken.As);

			var body = parser.ConsumeBeginBody();

			return new CreateFunctionSqlCodeExpr
			{
				Name = nameExpr,
				Arguments = arguments,
				ReturnVariable = returnVariableExpr,
				ReturnType = returnTypeExpr,
				Body = body
			};
		}

		private IExpression CreateProcedure(TextSpan token, IParser parser)
		{
			var nameExpr = parser.Scanner.ConsumeObjectId();
			var arguments = parser.ConsumeArgumentList();
			parser.Scanner.Consume(SqlToken.As);
			var bodyList = parser.ConsumeBeginBody();

			return new CreateProcedureSqlCodeExpr
			{
				Name = nameExpr,
				Arguments = arguments,
				Body = bodyList
			};
		}
	}
}