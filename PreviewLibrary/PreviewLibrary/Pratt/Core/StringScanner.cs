using PreviewLibrary.Pratt.TSql;
using PreviewLibrary.RecursiveParser;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PreviewLibrary.Pratt.Core
{
	public class StringScanner : IScanner
	{
		private Dictionary<string, int> _tokenMap = new Dictionary<string, int>();
		private ReadOnlyMemory<char> _textSpan;
		private int _index;

		public StringScanner(string text)
		{
			_textSpan = text.AsMemory();
			_index = -1;
		}

		public void AddToken(string token, int tokenType)
		{
			int maxTokenTypeValue = GetMaxTokenTypeValue() + 1;
			_tokenMap.Add(token, tokenType + maxTokenTypeValue);
		}

		private static int GetMaxTokenTypeValue()
		{
			var tokenTypeValues = (int[])Enum.GetValues(typeof(TokenType));
			var maxTokenTypeValue = tokenTypeValues.Last();
			return maxTokenTypeValue;
		}

		public string GetTokenTypeName<TTokenType>(int tokenTypeNumber)
		{
			int maxTokenTypeValue = GetMaxTokenTypeValue() + 1;
			try
			{
				var tokenType = Enum.ToObject(typeof(TokenType), tokenTypeNumber);
				return tokenType.ToString();
			}
			catch
			{
				var tokenType = Enum.ToObject(typeof(TTokenType), tokenTypeNumber - maxTokenTypeValue);
				return tokenType.ToString();
			}
		}

		public TextSpan Consume(string expect = null)
		{
			var token = ScanNext();
			if (!string.IsNullOrEmpty(expect))
			{
				var tokenStr = token.GetString(_textSpan.Span);
				if (!string.Equals(tokenStr, expect, StringComparison.OrdinalIgnoreCase))
				{
					throw new ScanException($"expect token '{expect}', but got '{tokenStr}'.");
				}
			}
			return token;
		}

		public string GetHelpMessage(TextSpan currentSpan)
		{
			var lnch = GetLineCh(currentSpan);
			var currentToken = GetSpanString(currentSpan);

			var sb = new StringBuilder();
			sb.AppendLine($"Line:{lnch.LineNumber} Ch:{lnch.ChNumber} CurrToken:'{currentToken}'");
			sb.AppendLine();

			sb.AppendLine(string.Join("\r\n", lnch.PrevLines));

			var line = lnch.Line.Replace("\t", " ");
			var spaces = new string(' ', line.Length);

			var down = new string('v', currentToken.Length);
			sb.AppendLine(spaces + down);
			sb.AppendLine(line + $"{currentToken}");
			var upper = new string('^', currentToken.Length);
			sb.AppendLine(spaces + upper);
			return sb.ToString();
		}

		public int GetOffset()
		{
			return _index;
		}

		public string GetSpanString(TextSpan span)
		{
			return span.GetString(_textSpan.Span);
		}

		public TextSpan Peek()
		{
			var startIndex = _index;
			var token = Consume();
			_index = startIndex;
			return token;
		}

		public void SetOffset(int offset)
		{
			_index = offset;
		}

		protected virtual TextSpan ScanNext()
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

			//if (character == '[' && TryRead(ReadSqlIdentifier, ch, out var sqlIdentifier))
			//{
			//	return sqlIdentifier;
			//}

			//if (character == '/' && TryRead(ReadMultiComment, ch, out var multiComment))
			//{
			//	return multiComment;
			//}

			//if (character == '-' && TryRead(ReadSingleComment, ch, out var signleComment))
			//{
			//	return signleComment;
			//}

			//if (character == '@' && TryRead(ReadVariable, ch, out var variable))
			//{
			//	return variable;
			//}

			//return ReadSymbol(ch);
			return ch;
		}

		protected TextSpan ReadIdentifier(TextSpan head)
		{
			var token = ReadUntil(head, (ch) =>
			{
				return IsIdentifierBody(ch);
			});
			var tokenStr = GetSpanString(token).ToUpper();
			token.Type = (int)TokenType.Identifier;
			return token;
		}

		protected TextSpan ReadNumber(TextSpan head)
		{
			var token = ReadUntil(head, (ch) =>
			{
				return char.IsDigit(ch);
			});
			token.Type = (int)TokenType.Number;
			return token;
		}

		protected virtual int GetTokenType(string token, int defaultTokenType)
		{
			if (!_tokenMap.TryGetValue(token, out var tokenType))
			{
				tokenType = defaultTokenType;
			}
			return tokenType;
		}

		protected bool IsIdentifierHead(char ch)
		{
			return ch == '_' || char.IsLetter(ch);
		}

		protected bool IsIdentifierBody(char ch)
		{
			return ch == '_' || char.IsLetterOrDigit(ch);
		}

		protected TextSpan ReadMultiComment(TextSpan head)
		{
			if (PeekCh() != '*')
			{
				return TextSpan.Empty;
			}

			var content = ReadUntil(head, ch =>
			{
				if (ch != '*')
				{
					return true;
				}
				if (PeekCh(1) == '/')
				{
					return false;
				}
				return true;
			});

			var tail = ConsumeCharacters("*/");
			content = content.Concat(tail);
			content.Type = (int)TokenType.MultiComment;
			return content;
		}

		protected TextSpan ReadUntil(TextSpan head, Func<char, bool> predicate)
		{
			var token = head;
			do
			{
				var ch = PeekSpan();
				if (ch.IsEmpty)
				{
					break;
				}
				if (!predicate(ch.GetCh(_textSpan.Span, 0)))
				{
					break;
				}
				token = token.Concat(ch);
				NextChar();
			} while (true);
			return token;
		}

		protected TextSpan SkipWhiteSpaceAtFront()
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

		protected char PeekCh(int offset = 0)
		{
			var chSpan = PeekSpan(offset);
			if (chSpan.IsEmpty)
			{
				return char.MinValue;
			}
			return GetSpanString(chSpan)[0];
		}

		protected TextSpan ConsumeCharacters(string expect)
		{
			var expectLength = expect.Length;
			if (_index + expectLength >= _textSpan.Length)
			{
				throw new ScanException($"expect read {expectLength} length, but remaining {_textSpan.Length - _index} length.");
			}

			var span = new TextSpan
			{
				Offset = _index + 1,
				Length = expectLength,
			};
			var spanStr = GetSpanString(span);
			if (!string.Equals(spanStr, expect, StringComparison.OrdinalIgnoreCase))
			{
				throw new ScanException($"expect '{expect}', but got '{spanStr}'.");
			}

			_index += expectLength;
			return span;
		}

		protected bool TryRead(Func<TextSpan, TextSpan> readSpan,
			TextSpan head, out TextSpan token)
		{
			token = readSpan(head);
			return !token.IsEmpty;
		}

		protected TextSpan PeekSpan(int offset = 0)
		{
			if (_index + 1 + offset >= _textSpan.Length)
			{
				return TextSpan.Empty;
			}
			return new TextSpan
			{
				Offset = _index + 1 + offset,
				Length = 1,
			};
		}

		protected TextSpan NextChar()
		{
			if (_index + 1 >= _textSpan.Length)
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

		protected LineChInfo GetLineCh(TextSpan currentSpan)
		{
			if (currentSpan.IsEmpty)
			{
				return new LineChInfo
				{
					LineNumber = 0,
					ChNumber = 0,
					PrevLines = new string[0],
					Line = string.Empty,
				};
			}

			var content = _textSpan.ToString();
			var previewContent = content.Substring(0, currentSpan.Offset);
			var lines = previewContent.Split("\r\n");
			var line = lines[lines.Length - 1];
			var prevLines = lines.SkipLast(1).TakeLast(3).ToArray();
			return new LineChInfo
			{
				LineNumber = lines.Length,
				ChNumber = line.Length + 1,
				PrevLines = prevLines,
				Line = line
			};
		}
	}
}
