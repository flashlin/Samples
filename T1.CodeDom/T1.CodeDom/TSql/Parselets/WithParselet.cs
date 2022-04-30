using System.Collections.Generic;
using System.Linq;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class WithParselet : IPrefixParselet
	{
		public IExpression Parse(TextSpan token, IParser parser)
		{
			if (!parser.IsToken(SqlToken.LParen))
			{
				return ParseWithTable(token, parser);
			}

			var optionList = parser.ParseWithOptionItemList();

			return new WithOptionSqlCodeExpr
			{
				OptionList = optionList 
			};
		}

		private SqlCodeExpr ParseWithTable(TextSpan withSpan, IParser parser)
		{
			var items = new List<WithItemSqlCodeExpr>();
			do
			{
				var table = parser.ConsumeObjectId();

				var columns = new List<SqlCodeExpr>();
				if (parser.Scanner.Match(SqlToken.LParen))
				{
					do
					{
						var column = parser.ConsumeObjectId();
						columns.Add(column);
					} while (parser.Scanner.Match(SqlToken.Comma));
					parser.Scanner.Consume(SqlToken.RParen);
				}

				parser.Scanner.Consume(SqlToken.As);
				parser.Scanner.Consume(SqlToken.LParen);
				var innerExpr = parser.ParseExpIgnoreComment();
				parser.Scanner.Consume(SqlToken.RParen);

				items.Add(new WithItemSqlCodeExpr
				{
					Table = table,
					Columns = columns,
					InnerExpr = innerExpr
				});
			} while (parser.Scanner.Match(SqlToken.Comma));
			
			return new WithTableSqlCodeExpr
			{
				Items = items
			};
		}
	}

	public class WithOptionSqlCodeExpr : SqlCodeExpr
	{
		public override void WriteToStream(IndentStream stream)
		{
			stream.Write("WITH(");
			OptionList.Select(x => x.ToUpper()).WriteToStreamWithComma(stream);
			stream.Write(")");
		}

		public List<string> OptionList { get; set; }
	}
}