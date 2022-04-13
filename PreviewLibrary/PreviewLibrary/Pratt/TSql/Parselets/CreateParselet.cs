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
			var helpMessage = parser.Scanner.GetHelpMessage();
			throw new ParseException($"Parse CREATE Error, {helpMessage}");
		}

		private IExpression CreateProcedure(TextSpan token, IParser parser)
		{
			var nameExpr = parser.Scanner.ConsumeObjectId();

			var arguments = parser.ConsumeByDelimiter(SqlToken.Comma, () =>
			{
				if (!parser.TryConsume(SqlToken.Variable, out var varName))
				{
					return null;
				}

				var dataType = parser.ConsumeDataType();

				return new ArgumentSqlCodeExpr
				{
					Name = varName as SqlCodeExpr,
					DataType = dataType,
				};
			}).ToList();

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