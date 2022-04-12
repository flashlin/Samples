using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class SelectParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var columns = new List<SqlCodeExpr>();
			do
			{
				columns.Add(ParseColumnAs(parser));
			} while (parser.Match(SqlToken.Comma));

			var fromSourceList = new List<SqlCodeExpr>();
			if (parser.Scanner.TryConsume(SqlToken.From, out _))
			{
				fromSourceList = parser.ConsumeByDelimiter(SqlToken.Comma, () =>
					{
						var sourceExpr = parser.ParseExp() as SqlCodeExpr;
						var userWithOptions = new List<string>();

						if (parser.Scanner.Match(SqlToken.With))
						{
							parser.Scanner.Consume(SqlToken.LParen);
							var withOptions = new[]
							{
								SqlToken.NOLOCK
							};
							userWithOptions = parser.Scanner.ConsumeToStringListByDelimiter(SqlToken.Comma, withOptions)
								.ToList();
							parser.Scanner.Consume(SqlToken.RParen);
						}

						return new FromSourceSqlCodeExpr
						{
							Left = sourceExpr,
							Options = userWithOptions,
						} as SqlCodeExpr;
					}).ToList();
			}

			SqlCodeExpr whereExpr = null;
			if (parser.Scanner.TryConsume(SqlToken.Where, out _))
			{
				whereExpr = parser.ParseExp() as SqlCodeExpr;
			}

			return new SelectSqlCodeExpr
			{
				Columns = columns,
				FromSourceList = fromSourceList,
				WhereExpr = whereExpr
			};
		}

		protected SqlCodeExpr ParseColumnAs(IParser parser)
		{
			var name = parser.ParseExp() as SqlCodeExpr;

			TextSpan aliasNameToken;
			if (parser.Scanner.TryConsumeAny(out _, SqlToken.As))
			{
				aliasNameToken = parser.Scanner.ConsumeAny(SqlToken.SqlIdentifier, SqlToken.Identifier);
				return new ColumnSqlCodeExpr
				{
					Name = name,
					AliasName = parser.Scanner.GetSpanString(aliasNameToken),
				};
			}

			parser.Scanner.TryConsumeAny(out aliasNameToken, SqlToken.SqlIdentifier, SqlToken.Identifier);

			return new ColumnSqlCodeExpr
			{
				Name = name,
				AliasName = parser.Scanner.GetSpanString(aliasNameToken),
			};
		}
	}
}
