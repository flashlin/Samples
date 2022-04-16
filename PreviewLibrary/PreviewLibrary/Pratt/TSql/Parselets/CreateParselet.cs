using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
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

			if (parser.Scanner.Match(SqlToken.Function))
			{
				return CreateFunction(token, parser);
			}

			if (parser.Scanner.Match(SqlToken.Partition))
			{
				return CreatePartitionFunction(token, parser);
			}

			var helpMessage = parser.Scanner.GetHelpMessage();
			throw new ParseException($"Parse CREATE Error, {helpMessage}");
		}

		private IExpression CreatePartitionFunction(TextSpan token, IParser parser)
		{
			if (parser.Scanner.IsToken(SqlToken.Scheme))
			{
				return CreatePartitionScheme(token, parser);
			}

			parser.Scanner.Consume(SqlToken.Function);

			var name = parser.Scanner.ConsumeObjectId();

			parser.Scanner.Consume(SqlToken.LParen);
			var dataType = parser.ConsumeDataType();
			parser.Scanner.Consume(SqlToken.RParen);

			parser.Scanner.Consume(SqlToken.As);
			parser.Scanner.Consume(SqlToken.Range);

			var rangeType = string.Empty;
			if (parser.Scanner.TryConsumeAny(out var rangeTypeSpan, SqlToken.Left, SqlToken.Right))
			{
				rangeType = parser.Scanner.GetSpanString(rangeTypeSpan);
			}

			parser.Scanner.Consume(SqlToken.For);
			parser.Scanner.Consume(SqlToken.Values);

			var boundaryValueList = new List<SqlCodeExpr>();
			parser.Scanner.Consume(SqlToken.LParen);
			do
			{
				var boundaryValue = parser.ParseExpIgnoreComment();
				boundaryValueList.Add(boundaryValue);
			} while (parser.Scanner.Match(SqlToken.Comma));
			parser.Scanner.Consume(SqlToken.RParen);

			return new CreatePartitionFunctionSqlCodeExpr
			{
				Name = name,
				DataType = dataType,
				RangeType = rangeType,
				BoundaryValueList = boundaryValueList,
			};
		}

		private IExpression CreatePartitionScheme(TextSpan token, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.Scheme);
			var schemeName = parser.Scanner.ConsumeObjectId();

			parser.Scanner.Consume(SqlToken.As);
			parser.Scanner.Consume(SqlToken.Partition);

			var funcName = parser.Scanner.ConsumeObjectId();

			parser.Scanner.TryConsumeString(SqlToken.All, out var allToken);
			parser.Scanner.Consume(SqlToken.To);

			var groupNameList = new List<SqlCodeExpr>();
			parser.Scanner.Consume(SqlToken.LParen);
			do
			{
				groupNameList.Add(parser.ConsumePrimary());
			}while(parser.Scanner.Match(SqlToken.Comma));
			parser.Scanner.Consume(SqlToken.RParen);

			return new CreatePartitionSchemeSqlCodeExpr
			{
				SchemeName = schemeName,
				FuncName = funcName,
				AllToken = allToken,
				GroupNameList = groupNameList
			};
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