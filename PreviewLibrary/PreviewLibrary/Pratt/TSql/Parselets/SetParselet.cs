using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class SetParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var sqlOptions = new[]
			{
				SqlToken.ANSI_NULLS,
				SqlToken.ANSI_PADDING,
				SqlToken.ANSI_WARNINGS,
				SqlToken.ARITHABORT,
				SqlToken.CONCAT_NULL_YIELDS_NULL,
				SqlToken.QUOTED_IDENTIFIER,
				SqlToken.NUMERIC_ROUNDABORT,
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

			var onOffToken = parser.Scanner.ConsumeAny(SqlToken.On, SqlToken.Off);
			var onOffStr = parser.Scanner.GetSpanString(onOffToken);

			return new SetSqlCodeExpr
			{
				Options = setOptions,
				Toggle = onOffStr,
			};
		}
	}
}
