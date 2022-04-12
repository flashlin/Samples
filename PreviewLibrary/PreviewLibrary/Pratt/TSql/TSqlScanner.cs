using PreviewLibrary.Pratt.Core;
using System;
using System.Text;

namespace PreviewLibrary.Pratt.TSql
{
	public class TSqlScanner : StringScanner
	{
		public TSqlScanner(string text)
			: base(text)
		{
			AddToken("AS", SqlToken.As);
			AddToken("AND", SqlToken.And);
			AddToken("BREAK", SqlToken.Break);
			AddToken("BEGIN", SqlToken.Begin);
			AddToken("ERROR", SqlToken.Error);
			AddToken("EXIT", SqlToken.Exit);
			AddToken("END", SqlToken.End);
			AddToken("EXISTS", SqlToken.Exists);
			AddToken("EXEC", SqlToken.Exec);
			AddToken("FROM", SqlToken.From);
			AddToken("GRANT", SqlToken.Grant);
			AddToken("IF", SqlToken.If);
			AddToken("INSERT", SqlToken.Insert);
			AddToken("NOT", SqlToken.Not);
			AddToken("LIKE", SqlToken.Like);
			AddToken("ON", SqlToken.On);
			AddToken("OFF", SqlToken.Off);
			AddToken("OR", SqlToken.Or);
			AddToken("GO", SqlToken.Go);
			AddToken("SET", SqlToken.Set);
			AddToken("SELECT", SqlToken.Select);
			AddToken("TO", SqlToken.To);
			AddToken("WHERE", SqlToken.Where);
			AddToken("WITH", SqlToken.With);
			AddToken(":SETVAR", SqlToken.ScriptSetVar);
			AddToken(":ON", SqlToken.ScriptOn);

			AddToken("ANSI_NULLS", SqlToken.ANSI_NULLS);
			AddToken("ANSI_PADDING", SqlToken.ANSI_PADDING);
			AddToken("ANSI_WARNINGS", SqlToken.ANSI_WARNINGS);
			AddToken("ARITHABORT", SqlToken.ARITHABORT);
			AddToken("CONNECT", SqlToken.CONNECT);
			AddToken("CONCAT_NULL_YIELDS_NULL", SqlToken.CONCAT_NULL_YIELDS_NULL);
			AddToken("QUOTED_IDENTIFIER", SqlToken.QUOTED_IDENTIFIER);
			AddToken("NUMERIC_ROUNDABORT", SqlToken.NUMERIC_ROUNDABORT);
			AddToken("NOEXEC", SqlToken.NOEXEC);
			AddToken("NOLOCK", SqlToken.NOLOCK);
			AddToken("IDENTITY_INSERT", SqlToken.IDENTITY_INSERT);

			AddSymbol("(", SqlToken.LParen);
			AddSymbol(")", SqlToken.RParen);
			AddSymbol(",", SqlToken.Comma);
			AddSymbol(";", SqlToken.Semicolon);
			AddSymbol(".", SqlToken.Dot);
			AddSymbol("=", SqlToken.Equal);
		}

		protected void AddToken(string token, SqlToken tokenType)
		{
			AddTokenMap(token.ToUpper(), tokenType.ToString());
		}

		protected void AddSymbol(string symbol, SqlToken tokenType)
		{
			AddSymbolMap(symbol, tokenType.ToString());
		}

		protected override string GetTokenType(string token, string defaultTokenType)
		{
			return base.GetTokenType(token.ToUpper(), defaultTokenType);
		}

		protected override bool TryScanNext(TextSpan headSpan, out TextSpan tokenSpan)
		{
			tokenSpan = TextSpan.Empty;

			var head = GetSpanString(headSpan);
			if (head == "N" && TryNextChar('\'', out var head2))
			{
				headSpan = headSpan.Concat(head2);
				if (!TryRead(ReadQuoteString, headSpan, out var nstring))
				{
					ThrowHelper.ThrowScanException(this, $"Scan NString Error.");
				}
				nstring.Type = SqlToken.NString.ToString();
				tokenSpan = nstring;
				return true;
			}

			if (head == "0" && TryNextChar('x', out var hexHead))
			{
				headSpan = headSpan.Concat(hexHead);
				if (!TryRead(ReadHexNumber, headSpan, out var hexString))
				{
					ThrowHelper.ThrowScanException(this, $"Scan NString Error.");
				}
				tokenSpan = hexString;
				return true;
			}


			if (head == "[" && TryRead(ReadSqlIdentifier, headSpan, out var sqlIdentifier))
			{
				tokenSpan = sqlIdentifier;
				return true;
			}

			if (head == ":" && TryRead(ReadIdentifier, headSpan, out var scriptIdentifier))
			{
				scriptIdentifier.Type = GetTokenType(scriptIdentifier, SqlToken.ScriptIdentifier);
				tokenSpan = scriptIdentifier;
				return true;
			}

			if (head == "/" && TryRead(ReadMultiComment, headSpan, out var multiComment))
			{
				tokenSpan = multiComment;
				return true;
			}

			if (head == "\"" && TryRead(ReadDoubleQuoteString, headSpan, out var doubleQuoteString))
			{
				tokenSpan = doubleQuoteString;
				return true;
			}

			if (head == "'" && TryRead(ReadQuoteString, headSpan, out var quoteString))
			{
				tokenSpan = quoteString;
				return true;
			}
			
			if (head == "@" && TryRead(ReadIdentifier, headSpan, out var variable))
			{
				tokenSpan = variable;
				tokenSpan.Type = SqlToken.Variable.ToString();
				return true;
			}


			return false;
		}

		private string GetTokenType(TextSpan span, SqlToken defaultTokenType)
		{
			var tokenStr = GetSpanString(span);
			return GetTokenType(tokenStr, defaultTokenType.ToString());
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
			content.Type = SqlToken.MultiComment.ToString();
			return content;
		}

		protected TextSpan ReadDoubleQuoteString(TextSpan head)
		{
			var content = ReadUntil(head, ch =>
			{
				return ch != '"';
			});

			var tail = ConsumeCharacters("\"");
			content = content.Concat(tail);
			content.Type = SqlToken.DoubleQuoteString.ToString();
			return content;
		}

		protected TextSpan ReadQuoteString(TextSpan head)
		{
			var prevCh = Char.MinValue;
			var content = ReadUntil(head, ch =>
			{
				var nextCh = PeekCh(1);
				if (ch == '\'' && nextCh == '\'')
				{
					prevCh = ch;
					return true;
				}
				if (ch == '\'' && prevCh == '\'')
				{
					prevCh = ch;
					return true;
				}
				prevCh = ch;
				return ch != '\'';
			});

			var tail = ConsumeCharacters("'");
			content = content.Concat(tail);
			content.Type = SqlToken.QuoteString.ToString();
			return content;
		}

		protected TextSpan ReadSqlIdentifier(TextSpan head)
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

			content = content.Concat(tail);
			content.Type = SqlToken.SqlIdentifier.ToString();
			return content;
		}
	}
}
