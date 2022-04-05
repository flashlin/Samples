

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
		public static readonly PrefixParselet Number =
			(token, parser) => new NumberSqlDom
			{
				Value = parser.GetSpanString(token)
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


		public static readonly PrefixParselet SelectNoFrom =
		  (token, parser) =>
		  {
			  var columns = ImmutableArray.CreateBuilder<SqlDom>();
			  do
			  {
				  columns.Add(parser.ParseExp(0));
			  } while (parser.Match(","));
			  //var conditionExpr = parser.ParseExp(0);
			  return new SelectNoFromSqlDom
			  {
				  Columns = columns
			  };
		  };
	}
}
