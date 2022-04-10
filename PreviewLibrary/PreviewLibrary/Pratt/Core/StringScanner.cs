using PreviewLibrary.Pratt.TSql;
using PreviewLibrary.RecursiveParser;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace PreviewLibrary.Pratt.Core
{
	public class StringScanner : IScanner
	{
		private Dictionary<string, string> _tokenToTokenTypeMap = new Dictionary<string, string>();
		private ReadOnlyMemory<char> _textSpan;
		private int _index;

		public StringScanner(string text)
		{
			_textSpan = text.AsMemory();
			_index = -1;
		}

		public void AddTokenMap(string token, string tokenType)
		{
			_tokenToTokenTypeMap.Add(token, tokenType);
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
			sb.AppendLine(line + $"{currentToken}{lnch.BackContent}");
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

			return ReadSymbol(ch);
		}

		private TextSpan ReadSymbol(TextSpan head)
		{
			var rgNotWord = new Regex(@"^\W$");
			var acc = new StringBuilder();
			acc.Append(GetSpanString(head));
			var token = ReadUntil(head, (ch) =>
			{
				if (char.IsWhiteSpace(ch))
				{
					return false;
				}

				//if (_tokenToTokenTypeMap.ContainsKey(acc.ToString()))
				//{
				//	return false;
				//}

				acc.Append($"{ch}");
				return rgNotWord.Match($"{ch}").Success;
			});
			var tokenStr = GetSpanString(token);
			token.Type = GetTokenType(tokenStr, TokenType.Symbol.ToString());
			return token;
		}

		protected TextSpan ReadIdentifier(TextSpan head)
		{
			var token = ReadUntil(head, (ch) =>
			{
				return IsIdentifierBody(ch);
			});

			var tokenStr = GetSpanString(token);
			token.Type = GetTokenType(tokenStr, TokenType.Identifier.ToString());
			return token;
		}

		protected TextSpan ReadNumber(TextSpan head)
		{
			var token = ReadUntil(head, (ch) =>
			{
				return char.IsDigit(ch);
			});
			token.Type = TokenType.Number.ToString();
			return token;
		}

		protected virtual string GetTokenType(string token, string defaultTokenType)
		{
			if (!_tokenToTokenTypeMap.TryGetValue(token, out var tokenType))
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

			var maxLen = content.Length - (currentSpan.Offset + currentSpan.Length);
			maxLen = Math.Min(maxLen, 50);

			var backContent = content.Substring(currentSpan.Offset + currentSpan.Length, maxLen);
			var backLine = backContent.Split("\r\n").First();

			return new LineChInfo
			{
				LineNumber = lines.Length,
				ChNumber = line.Length + 1,
				PrevLines = prevLines,
				Line = line,
				BackContent = backLine,
			};
		}
	}
}
