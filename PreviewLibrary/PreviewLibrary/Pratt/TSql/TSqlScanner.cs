using PreviewLibrary.RecursiveParser;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PreviewLibrary.Pratt.TSql
{
	public class StringScanner<TTokenType> : IScanner<TTokenType>
	{
		private Dictionary<string, TTokenType> _tokenMap = new Dictionary<string, TTokenType>();
		private ReadOnlyMemory<char> _textSpan;
		private int _index;

		public StringScanner(string text)
		{
			_textSpan = text.AsMemory();
			_index = -1;
		}

		public TextSpan<TTokenType> Consume(string expect=null)
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

		public string GetHelpMessage(TextSpan<TTokenType> currentSpan)
		{
			var lnch = GetLineCh(currentSpan);
			var currentToken = GetSpanString(currentSpan);

			var sb = new StringBuilder();
			sb.AppendLine($"Line:{lnch.LineNumber} Ch:{lnch.ChNumber} CurrToken:'{currentToken}'");
			sb.AppendLine();

			sb.AppendLine(string.Join("\r\n", lnch.PrevLines));

			var line = lnch.Line.Replace("\t", " ");
			var spaces = new String(' ', line.Length);

			var down = new String('v', currentToken.Length);
			sb.AppendLine(spaces + down);
			sb.AppendLine(line + $"{currentToken}");
			var upper = new String('^', currentToken.Length);
			sb.AppendLine(spaces + upper);
			return sb.ToString();
		}

		public int GetOffset()
		{
			return _index;
		}

		public string GetSpanString(TextSpan<TTokenType> span)
		{
			throw new NotImplementedException();
		}

		public TextSpan<TTokenType> Peek()
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

		protected virtual TextSpan<TTokenType> ScanNext()
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

			//if (char.IsDigit(character))
			//{
			//	return ReadNumber(ch);
			//}

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
			throw new InvalidOperationException();
		}

		protected TextSpan<TTokenType> ReadIdentifier(TextSpan<TTokenType> head)
		{
			var token = ReadUntil(head, (ch) =>
			{
				return IsIdentifierBody(ch);
			});
			var tokenStr = GetSpanString(token).ToUpper();
			//??
			//token.Type = GetTokenType(tokenStr);
			return token;
		}

		protected TTokenType GetTokenType(string token, TTokenType defaultTokenType)
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

		protected TextSpan<TTokenType> ReadMultiComment(TextSpan<TTokenType> head)
		{
			if (PeekCh() != '*')
			{
				return TextSpan<TTokenType>.Empty;
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
			//??
			//content.Type = SqlToken.MultiComment;
			return content;
		}

		protected TextSpan<TTokenType> ReadUntil(TextSpan<TTokenType> head, Func<char, bool> predicate)
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

		protected TextSpan<TTokenType> SkipWhiteSpaceAtFront()
		{
			TextSpan<TTokenType> ch;
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

		protected TextSpan<TTokenType> ConsumeCharacters(string expect)
		{
			var expectLength = expect.Length;
			if (_index + expectLength >= _textSpan.Length)
			{
				throw new ScanException($"expect read {expectLength} length, but remaining {_textSpan.Length - _index} length.");
			}

			var span = new TextSpan<TTokenType>
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

		protected bool TryRead(Func<TextSpan<TTokenType>, TextSpan<TTokenType>> readSpan, 
			TextSpan<TTokenType> head, out TextSpan<TTokenType> token)
		{
			token = readSpan(head);
			return !token.IsEmpty;
		}

		protected TextSpan<TTokenType> PeekSpan(int offset = 0)
		{
			if (_index + 1 + offset >= _textSpan.Length)
			{
				return TextSpan<TTokenType>.Empty;
			}
			return new TextSpan<TTokenType>
			{
				Offset = (_index + 1 + offset),
				Length = 1,
			};
		}

		protected TextSpan<TTokenType> NextChar()
		{
			if (_index + 1 >= _textSpan.Length)
			{
				return TextSpan<TTokenType>.Empty;
			}
			_index++;
			return new TextSpan<TTokenType>
			{
				Offset = _index,
				Length = 1
			};
		}

		protected LineChInfo GetLineCh(TextSpan<TTokenType> currentSpan)
		{
			if (currentSpan.IsEmpty)
			{
				return new LineChInfo
				{
					LineNumber = 0,
					ChNumber = 0,
					PrevLines = new string[0],
					Line = String.Empty,
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

	public class TSqlScanner : IScanner<SqlToken>
	{
		public TextSpan<SqlToken> Consume(string expect = null)
		{
			throw new NotImplementedException();
		}

		public string GetHelpMessage(TextSpan<SqlToken> currentSpan)
		{
			throw new NotImplementedException();
		}

		public int GetOffset()
		{
			throw new NotImplementedException();
		}

		public string GetSpanString(TextSpan<SqlToken> span)
		{
			throw new NotImplementedException();
		}

		public TextSpan<SqlToken> Peek()
		{
			throw new NotImplementedException();
		}

		public void SetOffset(int offset)
		{
			throw new NotImplementedException();
		}
	}
}
