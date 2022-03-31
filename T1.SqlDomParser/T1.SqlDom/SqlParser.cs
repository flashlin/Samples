using Superpower;
using Superpower.Model;
using Superpower.Parsers;
using Superpower.Tokenizers;
using System;

namespace T1.SqlDomParser
{
	public class SqlParser
	{
		public static Tokenizer<SqlToken> Tokenizer = new TokenizerBuilder<SqlToken>()
			.Ignore(Span.WhiteSpace)
			.Match(Character.EqualTo('+'), SqlToken.Plus)
			.Match(Character.EqualTo('-'), SqlToken.Minus)
			.Match(Character.EqualTo('*'), SqlToken.Times)
			.Match(Character.EqualTo('/'), SqlToken.Divide)
			.Match(Character.EqualTo('('), SqlToken.LParen)
			.Match(Character.EqualTo(')'), SqlToken.RParen)
			.Match(Numerics.Natural, SqlToken.Number)
			.Build();

		//static readonly TokenListParser<SqlToken, ParsedValue<Operators.Binary>> Add =
		//	Token.EqualTo(SqlToken.Plus).Select(t => Operators.Binary.Add.ToParsedValue(t.Span));

		//static readonly TokenListParser<SqlToken, ParsedValue<Operators.Binary>> Subtract =
		//	Token.EqualTo(SqlToken.Minus).Select(t => Operators.Binary.Sub.ToParsedValue(t.Span));

		public static TokenListParser<SqlToken, NumberLiteral> Number =
			  from s in (
					Token.EqualTo(SqlToken.Plus).Or(Token.EqualTo(SqlToken.Minus))
			  ).OptionalOrDefault(new Token<SqlToken>(SqlToken.Plus, TextSpan.None))
			  from n in Token.EqualTo(SqlToken.Number)
			  select new NumberLiteral(
					(n.ToStringValue(), s.Kind) switch
					{
						//(var v, SqlToken.Plus) when v.Equals("Inf", StringComparison.OrdinalIgnoreCase) => double.PositiveInfinity,
						//(var v, SqlToken.Minus) when v.Equals("Inf", StringComparison.OrdinalIgnoreCase) => double.NegativeInfinity,
						(var v, var op) => decimal.Parse(v) * (op == SqlToken.Minus ? -1 : 1)
					},
					s.Span.Length > 0 ? s.Span.UntilEnd(n.Span) : n.Span
			  );

		public static TokenListParser<SqlToken, SqlExpr> ExprNotBinary =
			from head in OneOf(
				 Parse.Ref(() => Number).Cast<SqlToken, NumberLiteral, SqlExpr>()
			)
			select head;

		static readonly TokenListParser<SqlToken, BinaryExpr> BinaryExpr =
			from head in Parse.Ref(() => ExprNotBinary)
			from tail in (
				 from opToken in Token.Matching<SqlToken>(x => BinaryOperatorMap!.ContainsKey(x), "binary_op")
				 let op = BinaryOperatorMap![opToken.Kind]
				 from expr in Parse.Ref(() => ExprNotBinary)
				 select (op, expr)
			).AtLeastOnce()
			select CreateBinaryExpression(head, tail);


		//static readonly TokenListParser<SqlExpressionToken, ExpressionType> Multiply =
		//	 Token.EqualTo(SqlExpressionToken.Times).Value(ExpressionType.MultiplyChecked);

		//static readonly TokenListParser<SqlExpressionToken, ExpressionType> Divide =
		//	 Token.EqualTo(SqlExpressionToken.Divide).Value(ExpressionType.Divide);

		public static TokenListParser<SqlToken, SqlExpr> Expr { get; } =
			from head in Parse.Ref(() => BinaryExpr).Cast<SqlToken, BinaryExpr, SqlExpr>().Try()
				.Or(ExprNotBinary)
			select head;


		public SqlExpr ParseSql(string expression)
		{
			//var expression = "1 * (2 + 3)";
			var tokenizer = Tokenizer;
			var tokenList = tokenizer.Tokenize(expression);

			var expressionTree = Expr.AtEnd().Parse(
				new TokenList<SqlToken>(tokenList.Where(x => x.Kind != SqlToken.Comment).ToArray())
				);
			return expressionTree;
		}

		private static BinaryExpr CreateBinaryExpression(SqlExpr head, (Operators.Binary op, SqlExpr expr)[] tail)
		{
			if (tail.Length == 1)
				return new BinaryExpr(head, tail[0].expr, tail[0].op, head.Span!.Value.UntilEnd(tail[0].expr.Span));

			var operands = new LinkedList<SqlExpr>(new[] { head }.Concat(tail.Select(x => x.expr)));
			var operators = new LinkedList<Operators.Binary>(tail.Select(x => x.op));

			foreach (var precedenceLevel in Operators.BinaryPrecedence)
			{
				var lhs = operands.First;
				var op = operators.First;

				// While we have operators left to consume, iterate through each operand + operator
				while (op != null)
				{
					var rhs = lhs!.Next!;

					// This operator has the same precedence of the current precedence level- create a new binary subexpression with the current operands + operators
					if (precedenceLevel.Contains(op.Value))
					{
						var b = new BinaryExpr(lhs.Value, rhs.Value, op.Value, lhs.Value.Span!.Value.UntilEnd(rhs.Value.Span));
						var bNode = operands.AddBefore(rhs, b);

						// Remove the previous operands (will replace with our new binary expression)
						operands.Remove(lhs);
						operands.Remove(rhs);

						lhs = bNode;
						var nextOp = op.Next;

						// Remove the operator
						operators.Remove(op);
						op = nextOp;
					}
					else
					{
						// Move on to the next operand + operator
						lhs = rhs;
						op = op.Next;
					}
				}
			}
			return (BinaryExpr)operands.Single();
		}

		private static TokenListParser<SqlToken, T> OneOf<T>(params TokenListParser<SqlToken, T>[] parsers)
		{
			TokenListParser<SqlToken, T> expr = parsers[0].Try();
			foreach (var p in parsers.Skip(1))
			{
				expr = expr.Or(p);
			}
			return expr;
		}

		private static readonly IReadOnlyDictionary<SqlToken, Operators.Binary> BinaryOperatorMap = new Dictionary<SqlToken, Operators.Binary>()
		{
			[SqlToken.Plus] = Operators.Binary.Add,
			[SqlToken.Minus] = Operators.Binary.Sub,
			[SqlToken.Times] = Operators.Binary.Mul,
		};
	}
}
