using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.TSql.Expressions;
using System;
using System.Collections.Generic;
using System.Linq;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql
{
	public static class SqlParserExtensions
	{
		public static bool Match(this IParser parser, SqlToken tokenType)
		{
			return parser.MatchTokenType(tokenType.ToString());
		}
	}

	public static class SqlCodeExprExtension
	{
		public static void WriteToStreamWithComma(this IEnumerable<SqlCodeExpr> exprList, IndentStream stream)
		{
			foreach (var expr in exprList.Select((val, idx) => new { val, idx }))
			{
				if (expr.idx != 0)
				{
					stream.Write(", ");
				}
				expr.val.WriteToStream(stream);
			}
		}

		public static void WriteToStreamWithComma(this IEnumerable<string> strList, IndentStream stream)
		{
			foreach (var str in strList.Select((val, idx) => new { val, idx }))
			{
				if (str.idx != 0)
				{
					stream.Write(", ");
				}
				stream.Write(str.val);
			}
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


		public static SqlCodeExpr ConsumeObjectId(this IScanner scanner)
		{
			return Consume(scanner, TryConsumeObjectId);
		}

		public delegate bool TryConsumeDelegate(IScanner scanner, out SqlCodeExpr expr);

		public static SqlCodeExpr Consume(this IScanner scanner, TryConsumeDelegate predicate)
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
