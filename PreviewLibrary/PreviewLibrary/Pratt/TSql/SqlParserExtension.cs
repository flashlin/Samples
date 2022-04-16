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
		public static SqlCodeExpr ConsumePrimary(this IParser parser)
		{
			if (parser.Scanner.TryConsumeAny(out var identifier, SqlToken.SqlIdentifier))
			{
				return parser.PrefixParse(identifier) as SqlCodeExpr;
			}
			parser.Scanner.Consume(SqlToken.Primary);
			return new ObjectIdSqlCodeExpr
			{
				ObjectName = "PRIMARY"
			};
		}


		public static IEnumerable<TExpression> ConsumeByDelimiter<TExpression>(this IParser parser,
			SqlToken delimiter,
			Func<TExpression> predicateExpr)
			where TExpression : SqlCodeExpr
		{
			parser.Scanner.IgnoreComments();
			return parser.ConsumeByDelimiter<SqlToken, TExpression>(delimiter, predicateExpr);
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

		public static List<SqlCodeExpr> ConsumeBeginBody(this IParser parser)
		{
			parser.Scanner.Consume(SqlToken.Begin);
			var bodyList = new List<SqlCodeExpr>();
			do
			{
				var body = parser.ParseExpIgnoreComment();
				if (body == null)
				{
					break;
				}
				bodyList.Add(body as SqlCodeExpr);
			} while (!parser.Scanner.TryConsume(SqlToken.End, out _));
			return bodyList;
		}

		public static SqlCodeExpr ConsumeDataType(this IParser parser)
		{
			var dataTypes = new[]
			{
				SqlToken.Bit,
				SqlToken.Bigint,
				SqlToken.Char,
				SqlToken.Date,
				SqlToken.DateTime,
				SqlToken.DateTime2,
				SqlToken.Decimal,
				SqlToken.Float,
				SqlToken.Int,
				SqlToken.Numeric,
				SqlToken.NVarchar,
				SqlToken.SmallDateTime,
				SqlToken.TinyInt,
				SqlToken.Varchar
			};

			if (parser.Scanner.Match(SqlToken.Table))
			{
				return ConsumeDataTableType(parser);
			}

			var dataTypeToken = parser.Scanner.ConsumeAny(dataTypes);
			var dataTypeStr = parser.Scanner.GetSpanString(dataTypeToken);

			var isPrimaryKey = ParseIsPrimaryKey(parser);

			if (!parser.Scanner.Match(SqlToken.LParen))
			{
				return new DataTypeSqlCodeExpr
				{
					DataType = dataTypeStr,
					IsPrimaryKey = isPrimaryKey,
				};
			}

			var size = ParseSize(parser);

			int? scale = null;
			if (parser.Scanner.Match(SqlToken.Comma))
			{
				var scaleToken = parser.Scanner.Consume(SqlToken.Number);
				scale = int.Parse(parser.Scanner.GetSpanString(scaleToken));
			}
			parser.Scanner.Consume(SqlToken.RParen);

			isPrimaryKey = ParseIsPrimaryKey(parser);

			return new DataTypeSqlCodeExpr
			{
				DataType = dataTypeStr,
				IsPrimaryKey = isPrimaryKey,
				Size = size,
				Scale = scale
			};
		}

		private static int? ParseSize(IParser parser)
		{
			int? size = null;
			if (parser.Scanner.Match(SqlToken.Max))
			{
				size = int.MaxValue;
			}
			else
			{
				var sizeToken = parser.Scanner.Consume(SqlToken.Number);
				var sizeStr = parser.Scanner.GetSpanString(sizeToken);
				size = int.Parse(sizeStr);
			}
			return size;
		}

		private static bool ParseIsPrimaryKey(IParser parser)
		{
			var isPrimaryKey = false;
			if (parser.Scanner.IsTokenList(SqlToken.Primary, SqlToken.Key))
			{
				parser.Scanner.Consume(SqlToken.Primary);
				parser.Scanner.Consume(SqlToken.Key);
				isPrimaryKey = true;
			}
			return isPrimaryKey;
		}

		public static Func<SqlCodeExpr> GetParseExpIgnoreCommentFunc(this IParser parser, int ctxPrecedence=0)
		{
			var comments = new List<CommentSqlCodeExpr>();
			return () =>
			{
				SqlCodeExpr expr = null;
				while (true)
				{
					expr = parser.ParseExp(ctxPrecedence) as SqlCodeExpr;
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

		public static List<TextSpan> IgnoreComments(this IScanner scanner)
		{
			var commentTypes = new[]
			{
				SqlToken.SingleComment.ToString(),
				SqlToken.MultiComment.ToString(),
			};
			var comments = new List<TextSpan>();
			do
			{
				var span = scanner.Peek();
				if (commentTypes.Contains(span.Type))
				{
					scanner.Consume();
					comments.Add(span);
					continue;
				}
				break;
			} while (true);
			return comments;
		}

		public static bool Match(this IParser parser, SqlToken tokenType)
		{
			return parser.Scanner.Match(tokenType);
		}

		public static SqlCodeExpr ParseExpIgnoreComment(this IParser parser, int ctxPrecedence=0)
		{
			return parser.GetParseExpIgnoreCommentFunc(ctxPrecedence)();
		}


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

		public static bool TryConsumeAliasName(this IParser parser, out SqlCodeExpr aliasNameExpr)
		{
			if (parser.Scanner.TryConsumeObjectId(out aliasNameExpr))
			{
				return true;
			}

			if (parser.Scanner.Match(SqlToken.As))
			{
				return parser.Scanner.TryConsumeObjectId(out aliasNameExpr);
			}

			aliasNameExpr = null;
			return false;
		}

		public static List<string> ParseWithOptions(this IParser parser)
		{
			var userWithOptions = new List<string>();
			if (parser.Scanner.Match(SqlToken.With))
			{
				parser.Scanner.Consume(SqlToken.LParen);
				var withOptions = new[]
				{
					SqlToken.NOLOCK,
					SqlToken.ROWLOCK,
				};
				userWithOptions = parser.Scanner.ConsumeToStringListByDelimiter(SqlToken.Comma, withOptions)
					.ToList();
				parser.Scanner.Consume(SqlToken.RParen);
			}
			return userWithOptions;
		}

		public static SqlCodeExpr PrefixParseAny(this IParser parser, int ctxPrecedence, params SqlToken[] prefixTokenTypeList)
		{
			var prefixTokenTypeStrList = prefixTokenTypeList.Select(x => x.ToString()).ToArray();
			for(var i=0; i < prefixTokenTypeList.Length; i++)
			{
				var prefixTokenType = prefixTokenTypeList[i];
				var prefixToken = parser.Scanner.Consume();
				if(!prefixTokenTypeStrList.Contains(prefixToken.Type))
				{
					ThrowHelper.ThrowParseException(parser, "");
				}
				return parser.PrefixParse(prefixToken, ctxPrecedence) as SqlCodeExpr;
			}
			throw new ParseException("");
		}
	}
}
