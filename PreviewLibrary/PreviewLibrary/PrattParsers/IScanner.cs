using System;
using System.Buffers;
using System.Collections.Generic;
using System.Text;


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
using System.Collections.Immutable;
using PreviewLibrary.PrattParsers.Expressions;

namespace PreviewLibrary.PrattParsers
{
	public interface IScanner
	{
		TextSpan Peek();
		TextSpan Consume(string expect = null);
		string GetSpanString(TextSpan span);
		bool Match(string expect);
		bool MatchIgnoreCase(string expect);
	}

	public interface IParser
	{
		SqlDom ParseProgram();
		SqlDom ParseExp(int ctxPrecedence);
		bool Match(string expect);
		void Consume(string expect);
	}

	public static class Parselets
	{
		public static readonly PrefixParselet Group =
			  (token, parser) =>
			  {
				  var expression = parser.ParseExp(0);
				  parser.Consume(")");
				  return expression;
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
	}

	static class Precedence
	{
		// Ordered in increasing precedence.
		public const int Assignment = 1;
		public const int Conditional = 2;
		public const int Sum = 3;
		public const int Product = 4;
		public const int Exponent = 5;
		public const int Prefix = 6;
		public const int Postfix = 7;
		public const int Call = 8;
	}

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

	public class SqlParser : IParser
	{
		private readonly IScanner _scanner;

		public SqlParser(IScanner scanner)
		{
			this._scanner = scanner;
		}

		public bool Match(string expect)
		{
			return _scanner.MatchIgnoreCase(expect);
		}

		public void Consume(string expect)
		{
			_scanner.Consume(expect);
		}

		public SqlDom ParseExp(int ctxPrecedence)
		{
			var prefixToken = _scanner.Consume();
			if (prefixToken.IsEmpty)
			{
				throw new Exception($"expect token but found none");
			}

			var prefixTokenStr = _scanner.GetSpanString(prefixToken);
			var prefixParse = SqlSpec.Instance.Prefix(prefixTokenStr);

			var left = prefixParse(prefixToken, this);

			while (true)
			{
				var infixToken = _scanner.Peek();
				if (infixToken.IsEmpty)
				{
					break;
				}

				var infixTokenStr = _scanner.GetSpanString(infixToken);
				var infixParselet = SqlSpec.Instance.Infix(infixTokenStr);
				if (infixParselet.parse == null)
				{
					break;
				}
				if (infixParselet.precedence <= ctxPrecedence) break;
				_scanner.Consume();
				left = infixParselet.parse(infixToken, left, this);
			}
			return left;
		}

		public SqlDom ParseProgram()
		{
			return ParseExp(0);
		}
	}
}
