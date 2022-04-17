using PreviewLibrary.Pratt.Core;
using PreviewLibrary.Pratt.Core.Expressions;
using PreviewLibrary.Pratt.Core.Parselets;
using PreviewLibrary.Pratt.TSql.Expressions;

namespace PreviewLibrary.Pratt.TSql.Parselets
{
	public class NStringParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			var tokenStr = parser.Scanner.GetSpanString(token);
			return new NStringSqlCodeExpr
			{
				Value = tokenStr
			};
		}
	}

	public class NullParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			return new NullSqlCodeExpr();
		}
	}


	public class TargetParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.Dot);
			var identifier = parser.PrefixParse(SqlToken.Identifier) as SqlCodeExpr;
			return new TargetSqlCodeExpr
			{
				Column = identifier
			};
		}
	}


	public class SourceParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.Consume(SqlToken.Dot);
			var identifier = parser.PrefixParse(SqlToken.Identifier) as SqlCodeExpr;
			return new SourceSqlCodeExpr
			{
				Column = identifier
			};
		}
	}
}
