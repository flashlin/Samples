using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

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
			throw new ParseException($"Parse OBJECT:: fail.\r\n{helpMessage}");
		}
	}
	
	public class TypeParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			if (parser.Scanner.Match(SqlToken.ColonColon))
			{
				var id = parser.ConsumeObjectId();
				return new TypeIdSqlCodeExpr
				{
					Id = id
				};
			}

			token.Type = SqlToken.Identifier.ToString();
			return parser.PrefixParse(token);
		}
	}

	public class TypeIdSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("TYPE::");
			Id.WriteToStream(stream);
		}

		public SqlCodeExpr Id { get; set; }
	}

	public class RowNumberParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			parser.Scanner.ConsumeList(SqlToken.LParen, SqlToken.RParen);
			
			var overExpr = parser.Consume(SqlToken.Over) as OverSqlCodeExpr;
			
			return new RowNumberSqlCodeExpr
			{
				Over = overExpr
			};
		}
	}
}
