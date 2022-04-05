using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

namespace PreviewLibrary.PrattParsers
{
	public class StringScanner : IScanner
	{
		private Dictionary<string, SqlToken> _tokenMap = new Dictionary<string, SqlToken>()
		{
			{ "+", SqlToken.Plus },
			{ ">=", SqlToken.GreaterThanOrEqual },
		};

		private ReadOnlyMemory<char> _textSpan;
		private int _index;

		public StringScanner(string text)
		{
			_textSpan = text.AsMemory();
			_index = -1;
		}

		public string GetSpanString(TextSpan span)
		{
			return span.GetString(_textSpan.Span);
		}

		public TextSpan Consume(string expect)
		{
			var token = ScanNext();
			if (!string.IsNullOrEmpty(expect))
			{
				var tokenStr = token.GetString(_textSpan.Span);
				if (tokenStr != expect)
				{
					throw new Exception($"expect token '{expect}', but got '{tokenStr}'");
				}
			}
			return token;
		}

		public TextSpan Peek()
		{
			var startIndex = _index;
			var token = ScanNext();
			_index = startIndex;
			return token;
		}

		public bool Match(string expect)
		{
			var tokenStr = Peek().GetString(_textSpan.Span);
			return tokenStr == expect;
		}

		public bool MatchIgnoreCase(string expect)
		{
			var tokenStr = Peek().GetString(_textSpan.Span);
			return string.Equals(tokenStr, expect, StringComparison.OrdinalIgnoreCase);
		}

		private TextSpan ScanNext()
		{
			var ch = SkipWhiteSpaceAtFront();
			if (ch.IsEmpty)
			{
				return ch;
			}

			var character = ch.GetCh(_textSpan.Span, 0);
			if (IsIdentifierHead(character))
			{
				return ReadIdentifier(ch);
			}

			if (char.IsDigit(character))
			{
				return ReadNumber(ch);
			}

			return ReadSymbol(ch);
		}

		private TextSpan ReadSymbol(TextSpan head)
		{
			var rg = new Regex(@"^\S$");
			var token = ReadUntil(head, (ch) =>
			{
				return rg.Match($"{ch}").Success;
			});
			var tokenStr = GetSpanString(token);
			token.Type = _tokenMap[tokenStr];
			return token;
		}

		private TextSpan ReadNumber(TextSpan head)
		{
			var token = ReadUntil(head, (ch) =>
			{
				return char.IsDigit(ch);
			});
			token.Type = SqlToken.Number;
			return token;
		}

		private TextSpan ReadIdentifier(TextSpan head)
		{
			var token = ReadUntil(head, (ch) =>
			{
				return IsIdentifierBody(ch);
			});
			token.Type = SqlToken.Identifier;
			return token;
		}

		private TextSpan ReadUntil(TextSpan head, Func<char, bool> predicate)
		{
			var token = head;
			do
			{
				var ch = PeekChar();
				if (ch.IsEmpty)
				{
					break;
				}
				if (!predicate(ch.GetCh(_textSpan.Span, 0)))
				{
					break;
				}
				token = Concat(token, ch);
				NextChar();
			} while (true);
			return token;
		}

		private TextSpan Concat(TextSpan span0, TextSpan span1)
		{
			return new TextSpan
			{
				Offset = span0.Offset,
				Length = span0.Length + span1.Length
			};
		}

		private bool IsIdentifierHead(char ch)
		{
			return ch == '_' || char.IsLetter(ch);
		}

		private bool IsIdentifierBody(char ch)
		{
			return ch == '_' || char.IsLetterOrDigit(ch);
		}

		private TextSpan NextChar()
		{
			if (_index >= _textSpan.Length)
			{
				return TextSpan.Empty;
			}
			_index++;
			return new TextSpan
			{
				Offset = _index,
				Length = 1
			};
		}

		private TextSpan PeekChar()
		{
			if (_index + 1 >= _textSpan.Length)
			{
				return TextSpan.Empty;
			}
			return new TextSpan
			{
				Offset = (_index + 1),
				Length = 1,
			};
		}

		private TextSpan SkipWhiteSpaceAtFront()
		{
			TextSpan ch;
			do
			{
				ch = NextChar();
				if (ch.IsEmpty)
				{
					break;
				}
				if (!char.IsWhiteSpace(ch.GetCh(_textSpan.Span, 0)))
				{
					break;
				}
			} while (true);
			return ch;
		}
	}
}
