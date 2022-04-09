using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;
using System.Collections.Generic;
using System.Linq;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class ObjectIdParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var tokenStr = parser.Scanner.GetSpanString(token);

			var identTokens = new List<TextSpan>();
			identTokens.Add(token);
			while (parser.Match(SqlToken.Dot))
			{
				if (identTokens.Count >= 3)
				{
					var prevTokens = string.Join(".", identTokens.Select(x => parser.Scanner.GetSpanString(x)));
					var currTokenStr = parser.Scanner.PeekString();
					throw new ParseException($"Expect Identifier.Identifier.Identifier, but got too many Identifier at '{prevTokens}.{currTokenStr}'.");
				}
				var identToken = parser.Scanner.ConsumeAny(SqlToken.SqlIdentifier, SqlToken.Identifier);
				identTokens.Add(identToken);
			}

			var fixCount = 3 - identTokens.Count;
			for (var i = 0; i < fixCount; i++)
			{
				identTokens.Add(TextSpan.Empty);
			}

			var identStr = identTokens.Select(x => parser.Scanner.GetSpanString(x)).ToList();

			return new ObjectIdSqlCodeExpr
			{
				DatabaseName = identStr[2],
				SchemaName = identStr[1],
				ObjectName = identStr[0],
			};
		}
	}
}
