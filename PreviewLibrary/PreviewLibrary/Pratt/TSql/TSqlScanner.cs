using PreviewLibrary.Pratt.Core;
using System;

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

			if (head == "/*" && TryRead(ReadMultiComment, span, out var multiComment))
			{
				return multiComment;
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
