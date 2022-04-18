using PreviewLibrary.Pratt.TSql.Expressions;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;

namespace T1.CodeDom.TSql.Parselets
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
