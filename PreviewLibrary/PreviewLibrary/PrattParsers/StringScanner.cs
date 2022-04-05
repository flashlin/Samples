using System;
using System.Text.RegularExpressions;

namespace PreviewLibrary.PrattParsers
{
	public class StringScanner : IScanner
	{
		private ReadOnlyMemory<char> _textSpan;
		private int _index;

		public StringScanner(string text)
		{
			_textSpan = text.AsMemory();
			_index = -1;
		}

		public ReadOnlySpan<char> Consume(string expect)
		{
			var token = ScanNext();
			if (expect != null && token != expect)
			{
				throw new Exception($"expect token '{expect}', but got '{token.ToString()}'");
			}
			return token;
		}

		public ReadOnlySpan<char> Peek()
		{
			var startIndex = _index;
			var token = ScanNext();
			_index = startIndex;
			return token;
		}

		private ReadOnlySpan<char> ScanNext()
		{
			var ch = SkipWhiteSpaceAtFront();
			if (ch.IsEmpty)
			{
				return ch;
			}

			if (IsIdentifierHead(ch))
			{
				return ReadIdentifier(ch);
			}

			if (char.IsDigit(ch[0]))
			{
				return ReadNumber(ch);
			}

			return ReadSymbol(ch);
		}

		private ReadOnlySpan<char> ReadSymbol(ReadOnlySpan<char> head)
		{
			var rg = new Regex(@"^\W$");
			return ReadUntil(head, (ch) =>
			{
				return rg.Match($"{ch}").Success;
			});
		}

		private ReadOnlySpan<char> ReadNumber(ReadOnlySpan<char> head)
		{
			return ReadUntil(head, (ch) =>
			{
				return char.IsDigit(ch);
			});
		}

		private ReadOnlySpan<char> ReadIdentifier(ReadOnlySpan<char> head)
		{
			return ReadUntil(head, (ch) =>
			{
				return IsIdentifierBody(ch);
			});
		}

		private ReadOnlySpan<char> ReadUntil(ReadOnlySpan<char> head, Func<char, bool> predicate)
		{
			var token = head;
			do
			{
				var ch = PeekChar();
				if (ch.IsEmpty)
				{
					break;
				}
				if (!predicate(ch[0]))
				{
					break;
				}
				token = SpanTool.Concat(token, ch);
				NextChar();
			} while (true);
			return token;
		}

		private bool IsIdentifierHead(ReadOnlySpan<char> ch)
		{
			return ch[0] == '_' || char.IsLetter(ch[0]);
		}

		private bool IsIdentifierBody(char ch)
		{
			return ch == '_' || char.IsLetterOrDigit(ch);
		}

		private ReadOnlySpan<char> NextChar()
		{
			if (_index >= _textSpan.Length)
			{
				return ReadOnlySpan<char>.Empty;
			}
			_index++;
			return _textSpan.Slice(_index, 1).Span;
		}

		private ReadOnlySpan<char> PeekChar()
		{
			if (_index + 1 >= _textSpan.Length)
			{
				return ReadOnlySpan<char>.Empty;
			}
			return _textSpan.Slice(_index + 1, 1).Span;
		}

		private ReadOnlySpan<char> SkipWhiteSpaceAtFront()
		{
			var ch = ReadOnlySpan<char>.Empty;
			do
			{
				ch = NextChar();
				if (ch.IsEmpty)
				{
					break;
				}
				if (!char.IsWhiteSpace(ch[0]))
				{
					break;
				}
			} while (true);
			return ch;
		}
	}
}
