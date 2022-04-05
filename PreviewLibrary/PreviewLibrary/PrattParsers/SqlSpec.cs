using System.Collections.Generic;


using PrefixParselet = System.Func<
	PreviewLibrary.PrattParsers.TextSpan,
	PreviewLibrary.PrattParsers.IParser,
	PreviewLibrary.PrattParsers.SqlDom>;

using InfixParselet = System.Func<
	PreviewLibrary.PrattParsers.TextSpan,
	PreviewLibrary.PrattParsers.SqlDom,
	PreviewLibrary.PrattParsers.IParser,
	PreviewLibrary.PrattParsers.SqlDom>;
using System.Linq;
using System.Collections;

namespace PreviewLibrary.PrattParsers
{
	public sealed class SqlSpec : IEnumerable
	{
		public static readonly SqlSpec Instance = new SqlSpec
		{
			 // Register all of the parselets for the grammar.

			 // Register the ones that need special parselets.
			 //{ Name     , Parselets.Name },
			 //{ Assign   , Precedence.Assignment, Parselets.Assign },
			 //{ Question , Precedence.Conditional, Parselets.Conditional },
			 { "(", Parselets.Group },
			 { "(", Precedence.Call, Parselets.Call },

			 // Register the simple operator parselets.
			 //{ Plus , Parselets.PrefixOperator(Precedence.Prefix) },
			 //{ Minus, Parselets.PrefixOperator(Precedence.Prefix) },
			 //{ Tilde, Parselets.PrefixOperator(Precedence.Prefix) },
			 //{ Bang , Parselets.PrefixOperator(Precedence.Prefix) },

			 // For kicks, we'll make "!" both prefix and postfix, kind of like ++.
			 //{ Bang, Precedence.Postfix, Parselets.PostfixOperator },

			 //{ Plus,     Precedence.Sum     , Parselets.BinaryOperator(Precedence.Sum     , isRight: false) },
			 //{ Minus,    Precedence.Sum     , Parselets.BinaryOperator(Precedence.Sum     , isRight: false) },
			 //{ Asterisk, Precedence.Product , Parselets.BinaryOperator(Precedence.Product , isRight: false) },
			 //{ Slash,    Precedence.Product , Parselets.BinaryOperator(Precedence.Product , isRight: false) },
			 //{ Caret,    Precedence.Exponent, Parselets.BinaryOperator(Precedence.Exponent, isRight: true ) },
		};

		readonly Dictionary<string, PrefixParselet> _prefixes = new Dictionary<string, PrefixParselet>();
		readonly Dictionary<string, (int, InfixParselet)> _infixes = new Dictionary<string, (int, InfixParselet)>();

		void Add(string token, PrefixParselet prefix) =>
			_prefixes.Add(token, prefix);

		void Add(string token, int precedence, InfixParselet prefix) =>
			_infixes.Add(token, (precedence, prefix));

		IEnumerator IEnumerable.GetEnumerator() =>
			_prefixes.Cast<object>().Concat(_infixes.Cast<object>()).GetEnumerator();

		public PrefixParselet Prefix(string token) => _prefixes[token];

		public (int precedence, InfixParselet parse) Infix(string token)
		{
			if (!_infixes.TryGetValue(token, out var infix))
			{
				return default;
			}
			return infix;
		}
	}
}
