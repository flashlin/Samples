﻿using System.Collections.Generic;
using System.Linq;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
	public class ExecParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			SqlCodeExpr returnVariable = null;
			if (parser.Scanner.TryConsume(SqlToken.Variable, out var returnVariableSpan))
			{
				returnVariable = parser.PrefixParse(returnVariableSpan, int.MaxValue) as SqlCodeExpr;
				parser.Scanner.Consume(SqlToken.Equal);
			}

			SqlCodeExpr funcName = null;

			if (parser.Scanner.Match(SqlToken.LParen))
			{
				funcName = new GroupSqlCodeExpr
				{
					InnerExpr = parser.ParseExpIgnoreComment()
				};
				parser.Scanner.Consume(SqlToken.RParen);
			}
			else
			{
				funcName = parser.ConsumeAny(SqlToken.SqlIdentifier, SqlToken.Identifier) as SqlCodeExpr;
			}

			var isEnd = parser.IsToken(SqlToken.End);
			

			var parameters = new List<SqlCodeExpr>();
			if (!isEnd)
			{
				do
				{
					var name = parser.ParseExpIgnoreComment();
					if (name == null)
					{
						break;
					}

					var isOutput = parser.MatchTokenAny(SqlToken.Out, SqlToken.Output);
					parameters.Add(new ParameterSqlCodeExpr
					{
						Name = name,
						IsOutput = isOutput
					});
				} while (parser.Scanner.Match(SqlToken.Comma));
			}

			return new ExecSqlCodeExpr
			{
				ExecToken = "EXEC",
				ReturnVariable = returnVariable,
				Name = funcName,
				Parameters = parameters
			};
		}
	}
}