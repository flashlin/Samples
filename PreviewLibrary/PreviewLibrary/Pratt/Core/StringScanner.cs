using PreviewLibrary.Pratt.TSql;
using PreviewLibrary.Pratt.TSql.Parselets;
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
		private Dictionary<string, string> _symbolToTokenTypeMap = new Dictionary<string, string>();
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

		protected void AddSymbolMap(string symbol, string tokenType)
		{
			_symbolToTokenTypeMap.Add(symbol, tokenType);
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


		public TextSpan ConsumeTokenType(string expectTokenType)
		{
			var token = ScanNext();
			if (token.IsEmpty)
			{
				ThrowHelper.ThrowScanException(this, $"Expect scan '{expectTokenType}', but got NONE.");
			}
			if (token.Type != expectTokenType)
			{
				ThrowHelper.ThrowScanException(this, $"Expect scan {expectTokenType}, but got {token.Type}.");
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

		protected virtual bool TryScanNext(TextSpan head, out TextSpan tokenSpan)
		{
			tokenSpan = TextSpan.Empty;
			return false;
		}

		protected TextSpan ScanNext()
		{
			var headSpan = SkipWhiteSpaceAtFront();
			if (headSpan.IsEmpty)
			{
				return headSpan;
			}

			if (TryScanNext(headSpan, out var tokenSpan))
			{
				return tokenSpan;
			}

			var character = headSpan.GetCh(_textSpan.Span, 0);
			if (IsIdentifierHead(character))
			{
				return ReadIdentifier(headSpan);
			}

			if (char.IsDigit(character))
			{
				return ReadNumber(headSpan);
			}

			_index--;
			var symbols = _symbolToTokenTypeMap.Keys.OrderByDescending(x => x.Length).ToArray();
			for (var i = 0; i < symbols.Length; i++)
			{
				var symbol = symbols[i];
				if (TryNextString(symbol, out var symbolSpan))
				{
					symbolSpan.Type = GetSymbolType(symbol, TokenType.Symbol.ToString());
					return symbolSpan;
				}
			}

			throw new ScanException($"Scan to '{character}' Fail.");
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

		protected string GetSymbolType(string symbol, string defaultTokenType)
		{
			if (!_symbolToTokenTypeMap.TryGetValue(symbol, out var tokenType))
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

		protected bool TryNextChar(char expectCharacter, out TextSpan tokenSpan)
		{
			var chSpan = PeekSpan();
			if (chSpan.IsEmpty)
			{
				tokenSpan = TextSpan.Empty;
				return false;
			}
			var ch = GetSpanString(chSpan)[0];
			if (ch != expectCharacter)
			{
				tokenSpan = TextSpan.Empty;
				return false;
			}
			tokenSpan = chSpan;
			NextChar();
			return true;
		}

		protected bool TryNextString(string expectString, out TextSpan tokenSpan)
		{
			var peekSpan = PeekSpan(0, expectString.Length);
			if (peekSpan.IsEmpty)
			{
				tokenSpan = TextSpan.Empty;
				return false;
			}
			var peekString = GetSpanString(peekSpan);
			if (peekString != expectString)
			{
				tokenSpan = TextSpan.Empty;
				return false;
			}
			tokenSpan = peekSpan;
			_index += peekString.Length;
			return true;
		}

		protected TextSpan ConsumeCharacters(string expect)
		{
			var expectLength = expect.Length;
			if (_index + expectLength >= _textSpan.Length)
			{
				throw new ScanException($"expect read {expectLength} length, but remaining {_textSpan.Length - _index - 1} length.");
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

		protected TextSpan PeekSpan(int offset, int length)
		{
			var startIndex = _index + 1 + offset;
			if (startIndex >= _textSpan.Length)
			{
				return TextSpan.Empty;
			}

			var maxLength = _textSpan.Length - startIndex;
			if (length > maxLength)
			{
				return TextSpan.Empty;
			}

			return new TextSpan
			{
				Offset = startIndex,
				Length = length,
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
