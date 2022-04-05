using System.Collections.Generic;


using PrefixParselet = System.Func<
	PreviewLibrary.PrattParsers.TextSpan,
	PreviewLibrary.PrattParsers.IParser,
	PreviewLibrary.PrattParsers.Expressions.SqlDom>;

using InfixParselet = System.Func<
	PreviewLibrary.PrattParsers.TextSpan,
	PreviewLibrary.PrattParsers.Expressions.SqlDom,
	PreviewLibrary.PrattParsers.IParser,
	PreviewLibrary.PrattParsers.Expressions.SqlDom>;
using System.Linq;
using System.Collections;

namespace PreviewLibrary.PrattParsers
{
	public sealed class SqlSpec : IEnumerable
	{
		public static readonly SqlSpec Instance = new SqlSpec
		{
			 // Register the ones that need special parselets.
			 { SqlToken.Number, Parselets.Number },
			 { SqlToken.Identifier, Parselets.Identifier },
			 //{ Assign   , Precedence.Assignment, Parselets.Assign },
			 //{ Question , Precedence.Conditional, Parselets.Conditional },
			 { SqlToken.LParen, Parselets.Group },
			 { SqlToken.LParen, Precedence.Call, Parselets.Call },
			 { SqlToken.Select, Parselets.Select },
			 { SqlToken.As, Precedence.As, Parselets.As },

			 { SqlToken.Plus , Parselets.PrefixOperator(Precedence.Prefix) },
			 { SqlToken.Minus, Parselets.PrefixOperator(Precedence.Prefix) },
			 //{ Tilde, Parselets.PrefixOperator(Precedence.Prefix) },
			 //{ Bang , Parselets.PrefixOperator(Precedence.Prefix) },

			 // For kicks, we'll make "!" both prefix and postfix, kind of like ++.
			 //{ Bang, Precedence.Postfix, Parselets.PostfixOperator },

			 { SqlToken.Plus,		Precedence.Sum, Parselets.BinaryOperator(Precedence.Sum, isRight: false) },
			 { SqlToken.Minus,	Precedence.Sum, Parselets.BinaryOperator(Precedence.Sum, isRight: false) },
			 { SqlToken.StarSign, Precedence.Product, Parselets.BinaryOperator(Precedence.Product , isRight: false) },
			 { SqlToken.Slash,	Precedence.Product, Parselets.BinaryOperator(Precedence.Product , isRight: false) },
			 //{ Caret,    Precedence.Exponent, Parselets.BinaryOperator(Precedence.Exponent, isRight: true ) },
		};

		readonly Dictionary<SqlToken, PrefixParselet> _prefixes = new Dictionary<SqlToken, PrefixParselet>();
		readonly Dictionary<SqlToken, (int, InfixParselet)> _infixes = new Dictionary<SqlToken, (int, InfixParselet)>();

		void Add(SqlToken token, PrefixParselet prefix) =>
			_prefixes.Add(token, prefix);

		void Add(SqlToken token, int precedence, InfixParselet prefix) =>
			_infixes.Add(token, (precedence, prefix));

		IEnumerator IEnumerable.GetEnumerator() =>
			_prefixes.Cast<object>().Concat(_infixes.Cast<object>()).GetEnumerator();

		public PrefixParselet Prefix(SqlToken token) => _prefixes[token];

		public (int precedence, InfixParselet parse) Infix(SqlToken token)
		{
			if (!_infixes.TryGetValue(token, out var infix))
			{
				return default;
			}
			return infix;
		}
	}
}
