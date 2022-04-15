using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;

namespace PreviewLibrary.Pratt.TSql
{
	public static class SqlScannerExtension
	{
		public delegate bool TryConsumeDelegate(IScanner scanner, out SqlCodeExpr expr);

		public static bool TryConsume(this IScanner scanner, SqlToken tokenType, out TextSpan span)
		{
			scanner.IgnoreComments();
			return scanner.TryConsume<SqlToken>(tokenType, out span);
		}

		public static SqlCodeExpr ConsumeObjectId(this IScanner scanner)
		{
			return Consume(scanner, TryConsumeObjectId);
		}

		public static bool Match(this IScanner scanner, SqlToken tokenType)
		{
			scanner.IgnoreComments();
			return scanner.Match<SqlToken>(tokenType);
		}

		public static bool TryConsumeObjectId(this IScanner scanner, out SqlCodeExpr expr)
		{
			var identTokens = new List<string>();
			do
			{
				if (identTokens.Count >= 3)
				{
					var prevTokens = string.Join(".", identTokens);
					var currTokenStr = scanner.PeekString();
					throw new ScanException($"Expect Identifier.Identifier.Identifier, but got too many Identifier at '{prevTokens}.{currTokenStr}'.");
				}
				if (!scanner.TryConsumeAny(out var identifier, SqlToken.Identifier, SqlToken.SqlIdentifier))
				{
					break;
				}
				identTokens.Add(scanner.GetSpanString(identifier));
			} while (scanner.Match(SqlToken.Dot));

			if (identTokens.Count == 0)
			{
				expr = null;
				return false;
			}

			var fixCount = 3 - identTokens.Count;
			for (var i = 0; i < fixCount; i++)
			{
				identTokens.Insert(0, string.Empty);
			}

			var identExpr = new ObjectIdSqlCodeExpr
			{
				DatabaseName = identTokens[0],
				SchemaName = identTokens[1],
				ObjectName = identTokens[2],
			};

			expr = identExpr;
			return true;
		}

		public static TextSpan Consume(this IScanner scanner, SqlToken tokenType)
		{
			scanner.IgnoreComments();
			return scanner.Consume<SqlToken>(tokenType);
		}

		private static SqlCodeExpr Consume(this IScanner scanner, TryConsumeDelegate predicate)
		{
			if (!predicate(scanner, out var expr))
			{
				var currentToken = scanner.Peek();
				var helpMessage = scanner.GetHelpMessage(currentToken);
				throw new ScanException(helpMessage);
			}
			return expr;
		}
	}
}
