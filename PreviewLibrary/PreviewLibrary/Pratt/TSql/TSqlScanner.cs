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
			AddToken("BREAK", SqlToken.Break);
			AddToken("ANSI_NULLS", SqlToken.ANSI_NULLS);
			AddToken("ANSI_PADDING", SqlToken.ANSI_PADDING);
			AddToken("ANSI_WARNINGS", SqlToken.ANSI_WARNINGS);
			AddToken("ARITHABORT", SqlToken.ARITHABORT);
			AddToken("CONCAT_NULL_YIELDS_NULL", SqlToken.CONCAT_NULL_YIELDS_NULL);
			AddToken("QUOTED_IDENTIFIER", SqlToken.QUOTED_IDENTIFIER);
			AddToken("NUMERIC_ROUNDABORT", SqlToken.NUMERIC_ROUNDABORT);
			AddToken("ON", SqlToken.On);
			AddToken("OFF", SqlToken.Off);
			AddToken("GO", SqlToken.Go);
			AddToken("SET", SqlToken.Set);
			AddToken("SELECT", SqlToken.Select);
			AddToken(",", SqlToken.Comma);
			AddToken(";", SqlToken.Semicolon);
		}

		protected void AddToken(string token, SqlToken tokenType)
		{
			AddTokenMap(token.ToUpper(), tokenType.ToString());
		}

		protected override string GetTokenType(string token, string defaultTokenType)
		{
			return base.GetTokenType(token.ToUpper(), defaultTokenType);
		}

		protected override TextSpan ScanNext()
		{
			var span = base.ScanNext();
			if (span.IsEmpty)
			{
				return span;
			}

			var head = GetSpanString(span);
			if (head == "[" && TryRead(ReadSqlIdentifier, span, out var sqlIdentifier))
			{
				return sqlIdentifier;
			}

			if (head == ":" && TryRead(ReadIdentifier, span, out var scriptIdentifier))
			{
				return scriptIdentifier;
			}

			if (head == "/*" && TryRead(ReadMultiComment, span, out var multiComment))
			{
				return multiComment;
			}

			if (head == "\"" && TryRead(ReadDoubleQuoteString, span, out var doubleQuoteString))
			{
				return doubleQuoteString;
			}

			return span;
		}

		protected TextSpan ReadMultiComment(TextSpan head)
		{
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
			var prevChar = char.MinValue;
			var content = ReadUntil(head, ch =>
			{
				if (prevChar == '\\' && ch == '"')
				{
					prevChar = ch;
					return true;
				}
				prevChar = ch;
				return ch != '"';
			});

			var tail = ConsumeCharacters("\"");
			content = content.Concat(tail);
			content.Type = SqlToken.DoubleQuoteString.ToString();
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
