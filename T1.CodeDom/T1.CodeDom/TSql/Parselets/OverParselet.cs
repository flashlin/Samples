using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
    public class OverParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            parser.Scanner.Consume(SqlToken.LParen);

            PartitionBySqlCodeExpr partitionBy = null;
            if (parser.Scanner.Match(SqlToken.Partition))
            {
                var partitionColumnList = new List<SqlCodeExpr>();
                parser.Scanner.Consume(SqlToken.By);
                do
                {
                    partitionColumnList.Add(parser.ConsumeObjectId());
                } while (parser.Scanner.Match(SqlToken.Comma));

                partitionBy = new PartitionBySqlCodeExpr()
                {
                    ColumnList = partitionColumnList
                };
            }

            OrderBySqlCodeExpr orderBy = null;
            if (parser.MatchTokenList(SqlToken.Order, SqlToken.By))
            {
                var orderColumnList = new List<SortSqlCodeExpr>();
                do
                {
                    //var name = parser.ConsumeObjectId();
                    var name = parser.ParseExpIgnoreComment();
                    
                    parser.Scanner.TryConsumeAny(out var sortTokenSpan, SqlToken.Asc, SqlToken.Desc);
                    var sortToken = parser.Scanner.GetSpanString(sortTokenSpan);
                    orderColumnList.Add(new SortSqlCodeExpr
                    {
                        Name = name,
                        SortToken = sortToken
                    });
                } while (parser.Scanner.Match(SqlToken.Comma));
                orderBy = new OrderBySqlCodeExpr
                {
                    ColumnList = orderColumnList
                };
            }
			

            parser.Scanner.Consume(SqlToken.RParen);
			
            return new OverSqlCodeExpr
            {
                PartitionBy = partitionBy,
                OrderBy = orderBy 
            };
        }
    }
}