using PreviewLibrary.Pratt.Core;
using System;

namespace PreviewLibrary.Pratt.TSql
{
	public class TSqlScanner : StringScanner
	{
		public TSqlScanner(string text)
			: base(text)
		{
			AddToken("SELECT", SqlToken.Select);
		}

		protected void AddToken(string token, SqlToken tokenType)
		{
			base.AddToken(token.ToUpper(), (int)tokenType);
		}

		protected override int GetTokenType(string token, int defaultTokenType)
		{
			return base.GetTokenType(token.ToUpper(), defaultTokenType);
		}

		protected override TextSpan ScanNext()
		{
			var span = base.ScanNext();

			var head = GetSpanString(span)[0];
			if (head == '[' && TryRead(ReadSqlIdentifier, span, out var sqlIdentifier))
			{
				return sqlIdentifier;
			}

			return span;
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
