using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.TSql.Expressions;
using System;
using System.Collections.Generic;
using System.Linq;
using T1.Standard.IO;

namespace PreviewLibrary.Pratt.TSql
{
	public static class SqlParserExtension
	{
		public static bool TryConsumeVariable(this IScanner scanner, out VariableSqlCodeExpr sqlExpr)
		{
			if (!scanner.TryConsume(SqlToken.Variable, out var returnVariableSpan))
			{
				sqlExpr = null;
				return false;
			}

			sqlExpr = new VariableSqlCodeExpr
			{
				Name = scanner.GetSpanString(returnVariableSpan)
			};
			return true;
		}

		public static bool Match(this IParser parser, SqlToken tokenType)
		{
			return parser.MatchTokenType(tokenType.ToString());
		}

		public static void WriteToStream(this IEnumerable<SqlCodeExpr> exprList, IndentStream stream,
			Action<IndentStream> writeDelimiter = null)
		{
			if (writeDelimiter == null)
			{
				writeDelimiter = (stream1) => stream1.WriteLine();
			}

			foreach (var expr in exprList.Select((val, idx) => new { val, idx }))
			{
				if (expr.idx != 0)
				{
					writeDelimiter(stream);
				}
				expr.val.WriteToStream(stream);
			}
		}

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

		public static SqlCodeExpr ConsumeDataType(this IParser parser)
		{
			var dataTypes = new[]
			{
				SqlToken.Bit,
				SqlToken.DateTime,
				SqlToken.DateTime2,
				SqlToken.Decimal,
				SqlToken.Int,
				SqlToken.Numeric,
				SqlToken.NVarchar,
				SqlToken.SmallDateTime,
				SqlToken.Varchar
			};

			if (parser.Scanner.Match(SqlToken.Table))
			{
				return ConsumeDataTableType(parser);
			}

			var dataTypeToken = parser.Scanner.ConsumeAny(dataTypes);
			var dataTypeStr = parser.Scanner.GetSpanString(dataTypeToken);

			if (!parser.Scanner.Match(SqlToken.LParen))
			{
				return new DataTypeSqlCodeExpr
				{
					DataType = dataTypeStr
				};
			}

			var sizeToken = parser.Scanner.Consume(SqlToken.Number);
			var sizeStr = parser.Scanner.GetSpanString(sizeToken);
			var size = int.Parse(sizeStr);

			int? scale = null;
			if (parser.Scanner.Match(SqlToken.Comma))
			{
				var scaleToken = parser.Scanner.Consume(SqlToken.Number);
				scale = int.Parse(parser.Scanner.GetSpanString(scaleToken));
			}

			parser.Scanner.Consume(SqlToken.RParen);
			return new DataTypeSqlCodeExpr
			{
				DataType = dataTypeStr,
				Size = size,
				Scale = scale
			};
		}

		private static SqlCodeExpr ConsumeDataTableType(IParser parser)
		{
			parser.Scanner.Consume(SqlToken.LParen);

			var columnDataTypeList = new List<SqlCodeExpr>();
			do
			{
				var name = parser.Scanner.ConsumeString(SqlToken.Identifier);
				var dataType = parser.ConsumeDataType();
				columnDataTypeList.Add(new ColumnDefineSqlCodeExpr
				{
					Name = name,
					DataType = dataType
				});
			} while (parser.Scanner.Match(SqlToken.Comma));
			parser.Scanner.Consume(SqlToken.RParen);
			return new DataTableTypeSqlCodeExpr
			{
				Columns = columnDataTypeList
			};
		}

		public static SqlCodeExpr ConsumeObjectId(this IScanner scanner)
		{
			return Consume(scanner, TryConsumeObjectId);
		}

		public static List<SqlCodeExpr> ConsumeBeginBody(this IParser parser)
		{
			parser.Scanner.Consume(SqlToken.Begin);
			var bodyList = new List<SqlCodeExpr>();
			do
			{
				var body = parser.ParseExp();
				bodyList.Add(body as SqlCodeExpr);
			} while (!parser.Scanner.TryConsume(SqlToken.End, out _));
			return bodyList;
		}

		public delegate bool TryConsumeDelegate(IScanner scanner, out SqlCodeExpr expr);

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

		public static List<ArgumentSqlCodeExpr> ConsumeArgumentList(this IParser parser)
		{
			var arguments = parser.ConsumeByDelimiter(SqlToken.Comma, () =>
			{
				if (!parser.TryConsume(SqlToken.Variable, out var varName))
				{
					return null;
				}

				var dataType = parser.ConsumeDataType();

				SqlCodeExpr defaultValueExpr = null;
				if (parser.Scanner.Match(SqlToken.Equal))
				{
					defaultValueExpr = parser.ParseExp() as SqlCodeExpr;
				}

				return new ArgumentSqlCodeExpr
				{
					Name = varName as SqlCodeExpr,
					DataType = dataType,
					DefaultValueExpr = defaultValueExpr
				};
			});

			return arguments.ToList();
		}


		public static Func<SqlCodeExpr> GetParseExpIgnoreCommentFunc(this IParser parser)
		{
			var comments = new List<CommentSqlCodeExpr>();
			return () =>
			{
				SqlCodeExpr expr = null;
				while (true)
				{
					expr = parser.ParseExp(0) as SqlCodeExpr;
					if (expr is CommentSqlCodeExpr commentExpr)
					{
						comments.Add(commentExpr);
						continue;
					}
					expr.Comments = comments;
					break;
				}
				return expr;
			};
		}
	}
}
