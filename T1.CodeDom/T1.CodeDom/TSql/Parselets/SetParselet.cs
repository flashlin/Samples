using System.Collections.Generic;
using System.Linq;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class DisableParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var triggerExpr = SqlParserExtension.ConsumeTrigger(parser);
			return new DisableSqlCodeExpr
			{
				Expr = triggerExpr,
			};
		}
	}
	
	public class EnableParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var triggerExpr = SqlParserExtension.ConsumeTrigger(parser);
			return new EnableSqlCodeExpr
			{
				Expr = triggerExpr,
			};
		}
	}

	public class SetParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			if(parser.Scanner.TryConsume(SqlToken.LOCK_TIMEOUT, out var lockTimeoutSpan))
			{
				return SetLockTimeout(lockTimeoutSpan, parser);
			}

			if (parser.Scanner.IsToken(SqlToken.DEADLOCK_PRIORITY))
			{
				return Set_DEADLOCK_PRIORITY(parser);
			}

			if( parser.Scanner.IsToken(SqlToken.TRANSACTION))
			{
				return SetTransaction(parser);
			}

			if (parser.Scanner.TryConsumeAny(out var variableSpan, SqlToken.Variable, SqlToken.Identifier, SqlToken.SqlIdentifier))
			{
				return SetVariable(variableSpan, parser);
			}

			if (parser.Scanner.Match(SqlToken.IDENTITY_INSERT))
			{
				var objectId = parser.PrefixParseAny(SqlToken.Identifier, SqlToken.SqlIdentifier) as SqlCodeExpr;
				var toggle = parser.Scanner.ConsumeAny(SqlToken.ON, SqlToken.OFF);
				var toggleStr = parser.Scanner.GetSpanString(toggle);
				return new SetIdentityInsertSqlCodeExpr
				{
					ObjectId = objectId,
					Toggle = toggleStr
				};
			}

			var sqlOptions = new[]
			{
				SqlToken.ANSI_NULLS,
				SqlToken.ANSI_PADDING,
				SqlToken.ANSI_WARNINGS,
				SqlToken.ARITHABORT,
				SqlToken.CONCAT_NULL_YIELDS_NULL,
				SqlToken.QUOTED_IDENTIFIER,
				SqlToken.NUMERIC_ROUNDABORT,
				SqlToken.NOEXEC,
				SqlToken.NOCOUNT,
				SqlToken.XACT_ABORT
			};

			var setOptions = new List<string>();
			do
			{
				if (!parser.Scanner.TryConsumeAny(out var optionToken, sqlOptions))
				{
					break;
				}
				var optionStr = parser.Scanner.GetSpanString(optionToken);
				setOptions.Add(optionStr);
			} while (parser.Match(SqlToken.Comma));

			if (setOptions.Count == 0)
			{
				var expect = string.Join(",", sqlOptions.Select(x => x.ToString()));
				ThrowHelper.ThrowParseException(parser, $"Expect one of {expect}.");
			}

			var onOffToken = parser.Scanner.ConsumeAny(SqlToken.ON, SqlToken.OFF);
			var onOffStr = parser.Scanner.GetSpanString(onOffToken);

			return new SetSqlCodeExpr
			{
				Options = setOptions,
				Toggle = onOffStr,
			};
		}

		private SqlCodeExpr SetLockTimeout(TextSpan lockTimeoutSpan, IParser parser)
		{
			var timeoutPeriod = parser.ParseExpIgnoreComment();
			
			return new SetLockTimeoutSqlCodeExpr
			{
				TimeoutPeriod = timeoutPeriod,
			};
		}

		private SetTransactionIsolationLevelSqlCodeExpr SetTransaction(IParser parser)
		{
			parser.Scanner.ConsumeList(SqlToken.TRANSACTION, SqlToken.ISOLATION, SqlToken.LEVEL);

			var spanList = new List<TextSpan>();
			var optionList = new[]
			{
				new []{  SqlToken.READ, SqlToken.UNCOMMITTED },
				new []{  SqlToken.READ, SqlToken.COMMITTED },
				new []{ SqlToken.REPEATABLE, SqlToken.READ },
				new []{ SqlToken.SNAPSHOT },
				new []{  SqlToken.SERIALIZABLE}
			};

			if( !parser.Scanner.TryConsumeListAny(out spanList, optionList) )
			{
				ThrowHelper.ThrowParseException(parser, "TRANSACTION option");
			}

			var optionStr = string.Join(" ", spanList.Select(s => s.ToString()));

			return new SetTransactionIsolationLevelSqlCodeExpr
			{
				Option = optionStr,
			};
		}

		private SetDealockPrioritySqlCodeExpr Set_DEADLOCK_PRIORITY(IParser parser)
		{
			parser.Scanner.Consume(SqlToken.DEADLOCK_PRIORITY);
			var priority = parser.Scanner.ConsumeStringAny(SqlToken.LOW, SqlToken.HIGH, SqlToken.NORMAL);

			return new SetDealockPrioritySqlCodeExpr
			{
				Priority = priority,
			};
		}

		private IExpression SetVariable(TextSpan variableSpan, IParser parser)
		{
			var variableExpr = new VariableSqlCodeExpr
			{
				Name = parser.Scanner.GetSpanString(variableSpan),
			};

			var oper = parser.Scanner.ConsumeStringAny(SqlToken.Equal, SqlToken.PlusEqual);

			var valueExpr = parser.ParseExp() as SqlCodeExpr;
			valueExpr = parser.ParseLRParenExpr(valueExpr);

			return new SetVariableSqlCodeExpr
			{
				Name = variableExpr,
				Oper = oper,
				Value = valueExpr
			};
		}
	}
}
