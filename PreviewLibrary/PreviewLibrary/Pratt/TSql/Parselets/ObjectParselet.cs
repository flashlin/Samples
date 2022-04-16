using PreviewLibrary.Exceptions;
using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class ObjectParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			if (parser.Scanner.Match(SqlToken.ColonColon))
			{
				var id = parser.ConsumeObjectId();
				return new ObjectSqlCodeExpr
				{
					Id = id
				};
			}

			var helpMessage = parser.Scanner.GetHelpMessage();
			throw new ParseException($"Parse OBJECT fail.\r\n{helpMessage}");
		}
	}
}
