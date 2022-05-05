using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class MergeParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var intoToken = string.Empty;
			if (parser.Scanner.Match(SqlToken.Into))
			{
				intoToken = "INTO";
			}

			var targetTable = parser.ConsumeObjectId();
			var withOptions = parser.ParseWithOption();

			if (!parser.TryConsumeAliasName(out var targetTableAliasName) && parser.Scanner.IsTokenList(SqlToken.As, SqlToken.Target))
			{
				parser.Scanner.Consume(SqlToken.As);
				parser.Scanner.Consume(SqlToken.Target);
				targetTableAliasName = new ObjectIdSqlCodeExpr
				{
					ObjectName = "Target"
				};
			}

			parser.Scanner.Consume(SqlToken.Using);

			var tableSource = parser.ParseExpIgnoreComment();

			if (!parser.TryConsumeAliasName(out var tableSourceAliasName) && parser.Scanner.IsTokenList(SqlToken.As, SqlToken.Source))
			{
				parser.Scanner.Consume(SqlToken.As);
				parser.Scanner.Consume(SqlToken.Source);
				targetTableAliasName = new ObjectIdSqlCodeExpr
				{
					ObjectName = "Source"
				};
			}

			var sourceColumnList = new List<SqlCodeExpr>();
			if (parser.MatchToken(SqlToken.LParen))
			{
				sourceColumnList = parser.ParseColumnList();
				parser.ConsumeToken(SqlToken.RParen);
			}

			parser.Scanner.Consume(SqlToken.ON);

			var mergeSearchCondition = parser.ParseExpIgnoreComment();

			var whenList = new List<SqlCodeExpr>();
			do
			{
				SqlCodeExpr whenItem = GetWhenNotMatched(parser);
				if (whenItem == null)
				{
					whenItem = GetWhenMatched(parser);
					if (whenItem == null)
					{
						break;
					}
				}
				whenList.Add(whenItem);
			} while (true);
			
			var outputList = parser.ParseOutputListExpr();

			parser.Scanner.Consume(SqlToken.Semicolon);

			return new MergeSqlCodeExpr
			{
				IntoToken = intoToken,
				TargetTable = targetTable,
				TargetTableAliasName = targetTableAliasName,
				WithOptions = withOptions,
				TableSource = tableSource,
				SourceColumnList = sourceColumnList,
				TableSourceAliasName = tableSourceAliasName,
				OnMergeSearchCondition = mergeSearchCondition,
				WhenList = whenList,
				OutputList = outputList,
			};
		}

		private MergeUpdateSqlCodeExpr ParseMergeUpdate(IParser parser)
		{
			if (!parser.Scanner.Match(SqlToken.Update))
			{
				return null;
			}
			parser.ConsumeToken(SqlToken.Set);

			var updateSetList = new List<AssignSqlCodeExpr>();
			do
			{
				var leftExpr = parser.ConsumeObjectId(true);
				parser.ConsumeToken(SqlToken.Equal);
				var rightExpr = parser.ParseExpIgnoreComment();
				updateSetList.Add(new AssignSqlCodeExpr
				{
					Left = leftExpr,
					Oper = "=",
					Right = rightExpr
				});
			} while (parser.Scanner.Match(SqlToken.Comma));

			return new MergeUpdateSqlCodeExpr
			{
				SetList = updateSetList
			};
		}

		private MergeInsertSqlCodeExpr GetMergeInsert(IParser parser)
		{
			if (!parser.Scanner.Match(SqlToken.Insert))
			{
				return null;
			}

			var insertColumnList = new List<SqlCodeExpr>();
			if (parser.MatchToken(SqlToken.LParen))
			{
				do
				{
					var column = parser.ParseExpIgnoreComment();
					insertColumnList.Add(column);
				} while (parser.Scanner.Match(SqlToken.Comma));
				parser.Scanner.Consume(SqlToken.RParen);
			}

			parser.Scanner.Consume(SqlToken.Values);

			var sourceColumnList = new List<SqlCodeExpr>();
			parser.Scanner.Consume(SqlToken.LParen);
			do
			{
				var column = parser.ParseExpIgnoreComment();
				sourceColumnList.Add(column);
			} while (parser.Scanner.Match(SqlToken.Comma));
			parser.Scanner.Consume(SqlToken.RParen);

			return new MergeInsertSqlCodeExpr
			{
				ColumnList = insertColumnList,
				SourceColumnList = sourceColumnList
			};
		}

		private WhenMatchedSqlCodeExpr GetWhenMatched(IParser parser)
		{
			if (!parser.Scanner.IsTokenList(SqlToken.When, SqlToken.Matched))
			{
				return null;
			}

			parser.Scanner.Consume(SqlToken.When);
			parser.Scanner.Consume(SqlToken.Matched);

			SqlCodeExpr clauseSearchCondition = null;
			if (parser.Scanner.Match(SqlToken.And))
			{
				clauseSearchCondition = parser.ParseExpIgnoreComment();
			}

			parser.Scanner.Consume(SqlToken.Then);

			SqlCodeExpr mergeMatched = ParseMergeUpdate(parser);
			if (mergeMatched == null)
			{
				mergeMatched = parser.ParseExpIgnoreComment();
			}

			return new WhenMatchedSqlCodeExpr
			{
				ClauseSearchCondition = clauseSearchCondition,
				MergeMatched = mergeMatched
			};
		}


		private WhenNotMatchedSqlCodeExpr GetWhenNotMatched(IParser parser)
		{
			if (!parser.Scanner.IsTokenList(SqlToken.When, SqlToken.Not, SqlToken.Matched))
			{
				return null;
			}

			parser.Scanner.Consume(SqlToken.When);
			parser.Scanner.Consume(SqlToken.Not);
			parser.Scanner.Consume(SqlToken.Matched);

			var byTarget = false;
			if (parser.Scanner.Match(SqlToken.By))
			{
				parser.Scanner.Consume(SqlToken.Target);
				byTarget = true;
			}

			SqlCodeExpr clauseSearchCondition = null;
			if (parser.Scanner.Match(SqlToken.And))
			{
				clauseSearchCondition = parser.ParseExpIgnoreComment();
			}

			parser.Scanner.Consume(SqlToken.Then);

			SqlCodeExpr mergeNotMatched = GetMergeInsert(parser);
			if (mergeNotMatched == null)
			{
				mergeNotMatched = parser.ParseExpIgnoreComment();
			}

			return new WhenNotMatchedSqlCodeExpr
			{
				ClauseSearchCondition = clauseSearchCondition,
				ByTarget = byTarget,
				MergeNotMatched = mergeNotMatched
			};
		}
	}
}
