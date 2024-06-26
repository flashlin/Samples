﻿

using PrefixParselet = System.Func<
	PreviewLibrary.PrattParsers.TextSpan,
	PreviewLibrary.PrattParsers.IParser,
	PreviewLibrary.PrattParsers.Expressions.SqlDom>;

using InfixParselet = System.Func<
	PreviewLibrary.PrattParsers.TextSpan,
	PreviewLibrary.PrattParsers.Expressions.SqlDom,
	PreviewLibrary.PrattParsers.IParser,
	PreviewLibrary.PrattParsers.Expressions.SqlDom>;

using Parselet = System.Func<
	PreviewLibrary.PrattParsers.IParser,
	PreviewLibrary.PrattParsers.Expressions.SqlDom>;

using System.Collections.Immutable;
using PreviewLibrary.PrattParsers.Expressions;
using PreviewLibrary.Exceptions;

namespace PreviewLibrary.PrattParsers
{
	public static class Parselets
	{
		public static readonly PrefixParselet MultiComment =
			(token, parser) =>
			{
				return new MultiCommentSqlDom
				{
					Content = parser.GetSpanString(token),
				};
			};

		public static readonly PrefixParselet SingleComment =
			(token, parser) =>
			{
				return new SingleCommentSqlDom
				{
					Content = parser.GetSpanString(token),
				};
			};

		public static readonly PrefixParselet Number =
			(token, parser) =>
			{
				var number = new NumberSqlDom
				{
					Value = parser.GetSpanString(token)
				};

				if (parser.Match(SqlToken.Identifier))
				{
					return new AliasSqlDom
					{
						Left = number,
						AliasName = parser.ParseExp(0)
					};
				}

				return number;
			};

		public static readonly PrefixParselet Identifier =
			(token, parser) =>
			{
				var identifier = new IdentifierSqlDom
				{
					Value = parser.GetSpanString(token)
				};

				//if (parser.Match(SqlToken.Identifier))
				//{
				//	return new AliasSqlDom
				//	{
				//		Left = identifier,
				//		AliasName = parser.ParseExp(0)
				//	};
				//}

				return identifier;
			};

		public static readonly PrefixParselet DataType =
			(token, parser) =>
			{
				return new DataTypeSqlDom
				{
					DataType = parser.GetSpanString(token)
				};
			};

		public static readonly Parselet Parameter =
			(parser) =>
			{
				if (!parser.Match(SqlToken.Variable))
				{
					return null;
				}

				var token = parser.Consume();

				var variableName = new VariableSqlDom
				{
					Value = parser.GetSpanString(token)
				};

				var dataType = parser.ParseBy(SqlToken.DataType);

				parser.TryParseBy(Size, out var sizeExpr);

				return new ParameterSqlDom
				{
					Name = variableName,
					DataType = dataType,
					Size = sizeExpr
				};
			};

		public static readonly Parselet Size =
			(parser) =>
			{
				if (!parser.TryConsume(SqlToken.LParen, out _))
				{
					return null;
				}

				var sizeToken = parser.Consume(SqlToken.Number);
				var size = int.Parse(parser.GetSpanString(sizeToken));

				int? scale = null;
				if (parser.TryConsume(SqlToken.Comma, out _))
				{
					var scaleToken = parser.Consume(SqlToken.Number);
					scale = int.Parse(parser.GetSpanString(scaleToken));
				}

				parser.Consume(SqlToken.RParen);
				return new SizeSqlDom
				{
					Size = size,
					Scale = scale
				};
			};


		public static readonly PrefixParselet Variable =
			(token, parser) =>
			{
				var variableName = new VariableSqlDom
				{
					Value = parser.GetSpanString(token)
				};

				if (parser.TryParseBy<AliasSqlDom>(SqlToken.Identifier, out var aliasName))
				{
					aliasName.Left = variableName;
					return aliasName;
				}

				return variableName;
			};

		public static readonly PrefixParselet ObjectId =
			(token, parser) =>
			{
				if (parser.TryConsumes(out var tokens2,
						new[] { SqlToken.Dot },
						new[] { SqlToken.Identifier, SqlToken.SqlIdentifier },
						new[] { SqlToken.Dot },
						new[] { SqlToken.Identifier, SqlToken.SqlIdentifier }))
				{
					var databaseName = parser.GetSpanString(token);
					var schemaName = parser.GetSpanString(tokens2[1]);
					var objectName = parser.GetSpanString(tokens2[3]);
					return new ObjectIdSqlDom
					{
						DatabaseName = databaseName,
						SchemaName = schemaName,
						ObjectName = objectName
					};
				}

				if (parser.TryConsumes(out var tokens1,
					new[] { SqlToken.Dot },
					new[] { SqlToken.Identifier, SqlToken.SqlIdentifier }))
				{
					var schemaName = parser.GetSpanString(token);
					var objectName = parser.GetSpanString(tokens1[1]);
					return new ObjectIdSqlDom
					{
						DatabaseName = string.Empty,
						SchemaName = schemaName,
						ObjectName = objectName
					};
				}

				return new ObjectIdSqlDom
				{
					DatabaseName = string.Empty,
					SchemaName = string.Empty,
					ObjectName = parser.GetSpanString(token)
				};
			};

		public static PrefixParselet PrefixOperator(int precedence) =>
			(token, parser) => new PrefixSqlDom
			{
				ValueType = token.Type,
				Value = parser.ParseExp(precedence)
			};

		public static InfixParselet BinaryOperator(int precedence, bool isRight) =>
			(token, left, parser) =>
				new OperatorSqlDom
				{
					Left = left,
					OperType = token.Type,
					Oper = parser.GetSpanString(token),
					Right = parser.ParseExp(precedence - (isRight ? 1 : 0))
				};


		public static readonly PrefixParselet Group =
			  (token, parser) =>
			  {
				  var expression = parser.ParseExp(0);
				  parser.Consume(")");
				  return new GroupSqlDom
				  {
					  Inner = expression
				  };
			  };

		public static readonly InfixParselet Call =
			(token, left, parser) =>
			{
				var args = ImmutableArray.CreateBuilder<SqlDom>();
				if (!parser.Match(")"))
				{
					do
					{
						args.Add(parser.ParseExp(0));
					} while (parser.Match(","));
					parser.Consume(")");
				}
				return new CallSqlDom(left, args.ToImmutable());
			};

		public static readonly PrefixParselet Select =
			(token, parser) =>
			{
				var columns = ImmutableArray.CreateBuilder<SqlDom>();
				do
				{
					columns.Add(parser.ParseExp(0));
				} while (parser.Match(","));

				if (!parser.Match("FROM"))
				{
					return new SelectNoFromSqlDom
					{
						Columns = columns
					};
				}
				parser.Consume("FROM");

				var table = parser.ParseExp(0);

				return new SelectSqlDom
				{
					Columns = columns,
					From = table
				};
			};

		public static readonly PrefixParselet Update =
			(token, parser) =>
			{
				return new UpdateSqlDom
				{

				};
			};

		public static readonly InfixParselet As =
			(token, left, parser) =>
			{
				var aliasName = parser.ParseExp(0);
				return new AliasSqlDom
				{
					Left = left,
					AliasName = aliasName,
				};
			};

		public static readonly PrefixParselet SelectNoFrom =
		  (token, parser) =>
		  {
			  var columns = ImmutableArray.CreateBuilder<SqlDom>();
			  do
			  {
				  columns.Add(parser.ParseExp(0));
			  } while (parser.Match(","));
			  return new SelectNoFromSqlDom
			  {
				  Columns = columns
			  };
		  };

		public static readonly PrefixParselet Create =
		  (token, parser) =>
		  {
			  if (parser.Match(SqlToken.Procedure))
			  {
				  return CreateProcedure(token, parser);
			  }
			  throw parser.CreateParseException(token);
		  };

		public static readonly PrefixParselet CreateProcedure =
		  (token, parser) =>
		  {
			  var procToken = parser.Consume();
			  var createProcToken = token.Concat(procToken);

			  var procedureName = parser.ParseByAny(SqlToken.SqlIdentifier, SqlToken.Identifier);

			  var columns = ImmutableArray.CreateBuilder<SqlDom>();
			  do
			  {
				  columns.Add(parser.ParseBy(Parameter));

			  } while (parser.TryConsume(",", out _));

			  parser.Consume(SqlToken.As);
			  parser.Consume(SqlToken.Begin);

			  var body = parser.ParseExp(0);

			  parser.Consume(SqlToken.End);

			  return new CreateProcedureSqlDom
			  {
				  Name = procedureName,
				  Parameters = columns,
				  Body = body
			  };
		  };
	}
}
