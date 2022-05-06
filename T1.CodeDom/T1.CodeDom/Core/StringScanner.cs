using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using T1.CodeDom.TSql;

namespace T1.CodeDom.Core
{
	public class StringScanner : IScanner
	{
		private int _index;
		protected Dictionary<string, string> _symbolToTokenTypeMap = new Dictionary<string, string>();
		private ReadOnlyMemory<char> _textSpan;
		private Dictionary<string, string> _tokenToTokenTypeMap = new Dictionary<string, string>();
		private HashSet<string> _funcNames = new HashSet<string>();
		
		public StringScanner(string text)
		{
			_textSpan = text.AsMemory();
			_index = -1;
		}

		protected void AddTokenMap<TTokenType>(string token, TTokenType tokenType)
			where TTokenType : struct
		{
			_tokenToTokenTypeMap.Add(token, tokenType.ToString());
		}
		
		protected void AddFuncNameMap<TTokenType>(string token, TTokenType tokenType)
			where TTokenType : struct
		{
			AddTokenMap(token, tokenType);
			_funcNames.Add(token);
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

		public TextSpan Peek(int n = 0)
		{
			var startIndex = _index;
			TextSpan token = TextSpan.Empty;
			for (var i = 0; i < n + 1; i++)
			{
				token = Consume();
			}
			_index = startIndex;
			return token;
		}

		public void SetOffset(int offset)
		{
			_index = offset;
		}

		public bool IsSymbol(int n=0)
		{
			var span = Peek(n);
			return IsSymbol(span);
		}
		
		public bool IsSymbol(TextSpan span)
		{
			var spanStr = GetSpanString(span);
			return _symbolToTokenTypeMap.ContainsKey(spanStr);
		}
		
		public bool IsFuncName(string spanStr)
		{
			return _funcNames.Contains(spanStr);
		}

		protected void AddSymbolMap<TTokenType>(string symbol, TTokenType tokenType)
			where TTokenType : struct
		{
			_symbolToTokenTypeMap.Add(symbol, tokenType.ToString());
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
			var lines = previewContent.Split("\n").Select(x => x.Replace("\r","")).ToArray();
			var line = lines[lines.Length - 1];
			var prevLines = lines.SkipLast(1).TakeLast(3).ToArray();

			var maxLen = content.Length - (currentSpan.Offset + currentSpan.Length);
			maxLen = Math.Min(maxLen, 50);

			var backContent = content.Substring(currentSpan.Offset + currentSpan.Length, maxLen);
			var backLine = backContent.Split("\n").Select(x => x.Replace("\r", "")).First();

			return new LineChInfo
			{
				LineNumber = lines.Length,
				ChNumber = line.Length + 1,
				PrevLines = prevLines,
				Line = line,
				BackContent = backLine,
			};
		}

		protected string GetSymbolType(string symbol, string defaultTokenType)
		{
			if (!_symbolToTokenTypeMap.TryGetValue(symbol, out var tokenType))
			{
				tokenType = defaultTokenType;
			}
			return tokenType;
		}

		protected virtual string GetTokenType(string token, string defaultTokenType)
		{
			if (!_tokenToTokenTypeMap.TryGetValue(token, out var tokenType))
			{
				tokenType = defaultTokenType;
			}
			return tokenType;
		}

		protected bool IsIdentifierBody(char ch)
		{
			return ch == '_' || char.IsLetterOrDigit(ch);
		}

		protected bool IsIdentifierHead(char ch)
		{
			return ch == '_' || char.IsLetter(ch);
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

		protected char PeekCh(int offset = 0)
		{
			var chSpan = PeekSpan(offset);
			if (chSpan.IsEmpty)
			{
				return char.MinValue;
			}
			return GetSpanString(chSpan)[0];
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

		protected TextSpan ReadHexNumber(TextSpan head)
		{
			var token = ReadUntil(head, (ch) =>
			{
				return char.IsLetterOrDigit(ch);
			});

			var tokenStr = GetSpanString(token);
			token.Type = GetTokenType(tokenStr, SqlToken.HexNumber.ToString());
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

		protected TextSpan ReadUntil(TextSpan head, Func<char, bool> predicate)
		{
			var token = head;
			do
			{
				var charToken = PeekSpan();
				if (charToken.IsEmpty)
				{
					break;
				}
				if (!predicate(charToken.GetCh(_textSpan.Span, 0)))
				{
					break;
				}
				token = token.Concat(charToken);
				NextChar();
			} while (true);
			return token;
		}

		public TextSpan ScanNext()
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

			if (TryRead(ReadSymbol, headSpan, out var symbol))
			{
				return symbol;
			}

			var helpMessage = GetHelpMessage(headSpan);
			throw new ScanException($"Scan to '{character}' Fail.\r\n{helpMessage}");
		}

		protected TextSpan ReadSymbol(TextSpan headSpan)
		{
			var maxSymbolLen = _symbolToTokenTypeMap.Keys.OrderByDescending(x => x.Length).First().Length;
			for (var tailLen = maxSymbolLen - 1; tailLen >= 0; tailLen--)
			{
				var tailSpan = PeekSpan(0, tailLen);
				var span = headSpan.Concat(tailSpan);
				var spanStr = GetSpanString(span);
				if (_symbolToTokenTypeMap.TryGetValue(spanStr, out var symbolType))
				{
					_index += tailLen;
					span.Type = symbolType;
					return span;
				}
			}
			return TextSpan.Empty;
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

		protected bool TryRead(Func<TextSpan, TextSpan> readSpan,
			TextSpan head, out TextSpan token)
		{
			var startIndex = GetOffset();
			token = readSpan(head);
			if (token.IsEmpty)
			{
				SetOffset(startIndex);
			}
			return !token.IsEmpty;
		}

		protected virtual bool TryScanNext(TextSpan head, out TextSpan tokenSpan)
		{
			tokenSpan = TextSpan.Empty;
			return false;
		}
	}
}
