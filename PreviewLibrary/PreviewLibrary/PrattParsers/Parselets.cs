

using PrefixParselet = System.Func<
	PreviewLibrary.PrattParsers.TextSpan,
	PreviewLibrary.PrattParsers.IParser,
	PreviewLibrary.PrattParsers.Expressions.SqlDom>;

using InfixParselet = System.Func<
	PreviewLibrary.PrattParsers.TextSpan,
	PreviewLibrary.PrattParsers.Expressions.SqlDom,
	PreviewLibrary.PrattParsers.IParser,
	PreviewLibrary.PrattParsers.Expressions.SqlDom>;
using System.Collections.Immutable;
using PreviewLibrary.PrattParsers.Expressions;

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

				if (parser.Match(SqlToken.Identifier))
				{
					return new AliasSqlDom
					{
						Left = identifier,
						AliasName = parser.ParseExp(0)
					};
				}

				return identifier;
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

		//public static readonly PrefixParselet ObjectId = 
		//	(token, parser) =>
		//	{
		//		if(parser.Match(Sql))
		//	};


		public static readonly PrefixParselet Create =
		  (token, parser) =>
		  {
			  if(parser.Match(SqlToken.Procedure))
			  {
				  return CreateProcedure(token, parser);
			  }
			  throw new System.Exception("");
		  };

		public static readonly PrefixParselet CreateProcedure =
		  (token, parser) =>
		  {
			  var procToken = parser.Consume();
			  var createProcToken = token.Concat(procToken);


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
	}
}
