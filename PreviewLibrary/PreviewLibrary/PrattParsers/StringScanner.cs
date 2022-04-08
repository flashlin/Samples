using PreviewLibrary.Exceptions;
using PreviewLibrary.RecursiveParser;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace PreviewLibrary.PrattParsers
{
	public class StringScanner : IScanner
	{
		private Dictionary<string, SqlToken> _tokenMap = new Dictionary<string, SqlToken>()
		{
			{ "+", SqlToken.Plus },
			{ "-", SqlToken.Minus },
			{ "*", SqlToken.StarSign },
			{ "/", SqlToken.Slash },
			{ "(", SqlToken.LParen },
			{ ")", SqlToken.RParen },
			{ ".", SqlToken.Dot },
			{ ">=", SqlToken.GreaterThanOrEqual },
			{ "AS", SqlToken.As },
			{ "BEGIN", SqlToken.Begin },
			{ "CREATE", SqlToken.Create },
			{ "END", SqlToken.End },
			{ "FROM", SqlToken.From },
			{ "INT", SqlToken.DataType },
			{ "PROCEDURE", SqlToken.Procedure },
			{ "PROC", SqlToken.Procedure },
			{ "SELECT", SqlToken.Select },
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

		public int GetOffset()
		{
			return _index;
		}

		public void SetOffset(int offset)
		{
			_index = offset;
		}

		public TextSpan Consume(string expect)
		{
			var token = ScanNext();
			if (!string.IsNullOrEmpty(expect))
			{
				var tokenStr = token.GetString(_textSpan.Span);
				if (!string.Equals(tokenStr, expect, StringComparison.OrdinalIgnoreCase))
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

		public string PeekString()
		{
			var token = Peek();
			return GetSpanString(token);
		}

		public bool Match(string expect)
		{
			var tokenStr = Peek().GetString(_textSpan.Span);
			return string.Equals(tokenStr, expect, StringComparison.OrdinalIgnoreCase);
		}

		public bool Match(SqlToken expectToken)
		{
			var token = Peek();
			if ( token.IsEmpty )
			{
				return false;
			}
			return token.Type == expectToken;
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

			if (character == '[' && TryRead(ReadSqlIdentifier, ch, out var sqlIdentifier))
			{
				return sqlIdentifier;
			}

			if (character == '/' && TryRead(ReadMultiComment, ch, out var multiComment))
			{
				return multiComment;
			}

			if (character == '-' && TryRead(ReadSingleComment, ch, out var signleComment))
			{
				return signleComment;
			}

			if (character == '@' && TryRead(ReadVariable, ch, out var variable))
			{
				return variable;
			}

			return ReadSymbol(ch);
		}

		private TextSpan ReadVariable(TextSpan head)
		{
			var rgNonSpaces = new Regex(@"^\S$");
			var content = ReadUntil(head, ch =>
			{
				return rgNonSpaces.Match($"{ch}").Success;
			});
			content.Type = SqlToken.Variable;
			return content;
		}

		private bool TryRead(Func<TextSpan, TextSpan> readSpan, TextSpan head, out TextSpan token)
		{
			token = readSpan(head);
			return !token.IsEmpty;
		}

		private TextSpan ReadSqlIdentifier(TextSpan head)
		{
			if (PeekCh() == ']')
			{
				return TextSpan.Empty;
			}

			var content = ReadUntil(head, ch =>
			{
				return ch != ']';
			});

			var tail = ConsumeCharacters("]");

			content = Concat(content, tail);
			content.Type = SqlToken.SqlIdentifier;
			return content;
		}

		private TextSpan ReadMultiComment(TextSpan head)
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
			content = Concat(content, tail);
			content.Type = SqlToken.MultiComment;
			return content;
		}

		private TextSpan ReadSingleComment(TextSpan head)
		{
			if (PeekCh() != '-')
			{
				return TextSpan.Empty;
			}

			var content = ReadUntil(head, ch =>
			{
				return ch != '\r';
			});

			if (PeekCh() == '\r')
			{
				ConsumeCharacters("\r");
			}
			if (PeekCh() == '\n')
			{
				ConsumeCharacters("\n");
			}
			content.Type = SqlToken.SingleComment;
			return content;
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

				if (_tokenMap.ContainsKey(acc.ToString()))
				{
					return false;
				}

				acc.Append($"{ch}");
				return rgNotWord.Match($"{ch}").Success;
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
			var tokenStr = GetSpanString(token).ToUpper();
			token.Type = GetTokenTypeOrIdentifier(tokenStr);
			return token;
		}

		private SqlToken GetTokenTypeOrIdentifier(string token)
		{
			if (!_tokenMap.TryGetValue(token, out var tokenType))
			{
				tokenType = SqlToken.Identifier;
			}
			return tokenType;
		}

		private TextSpan ReadUntil(TextSpan head, Func<char, bool> predicate)
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

		public LineChInfo GetLineCh(TextSpan currentSpan)
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

		public string GetHelpMessage(TextSpan currentSpan)
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

		public ParseException CreateParseException(TextSpan currentSpan)
		{
			var tokenStr = GetSpanString(currentSpan);
			var message = GetHelpMessage(currentSpan);
			return new ParseException($"Prefix '{tokenStr}' Parse fail.\r\n" + message);
		}

		private TextSpan PeekSpan(int offset = 0)
		{
			if (_index + 1 + offset >= _textSpan.Length)
			{
				return TextSpan.Empty;
			}
			return new TextSpan
			{
				Offset = (_index + 1 + offset),
				Length = 1,
			};
		}

		private char PeekCh(int offset = 0)
		{
			var chSpan = PeekSpan(offset);
			if (chSpan.IsEmpty)
			{
				return char.MinValue;
			}
			return GetSpanString(chSpan)[0];
		}

		private TextSpan ConsumeCharacters(string expect)
		{
			var expectLength = expect.Length;
			if (_index + expectLength >= _textSpan.Length)
			{
				throw new Exception($"expect read {expectLength} length, but remaining {_textSpan.Length - _index}.");
			}

			var span = new TextSpan
			{
				Offset = _index + 1,
				Length = expectLength,
			};
			var spanStr = GetSpanString(span);
			if (!string.Equals(spanStr, expect, StringComparison.OrdinalIgnoreCase))
			{
				throw new Exception($"expect '{expect}', but got '{spanStr}'.");
			}

			_index += expectLength;
			return span;
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
