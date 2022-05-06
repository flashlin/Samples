using System;
using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
    public class DbccParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            parser.ConsumeToken(SqlToken.UpdateUsage);
            parser.ConsumeToken(SqlToken.LParen);
            var objectIdList = new List<SqlCodeExpr>();
            do
            {
                var objectId = parser.ParseExpIgnoreComment();
                objectIdList.Add(objectId);
            } while (parser.MatchToken(SqlToken.Comma));

            parser.ConsumeToken(SqlToken.RParen);

            var withList = new List<SqlCodeExpr>();
            if (parser.MatchToken(SqlToken.With))
            {
                withList = parser.ParseAll(ParseSqlTokenFn(SqlToken.NO_INFOMSGS),
                    ParseSqlTokenFn(SqlToken.COUNT_ROWS));
            }

            return new DbccUpdateusageSqlCodeExpr
            {
                ObjectIdList = objectIdList,
                WithList = withList,
            };
        }


        private Func<IParser, SqlCodeExpr> ParseSqlTokenFn(SqlToken sqlToken)
        {
            return (IParser parser) =>
            {
                if (!parser.MatchToken(sqlToken))
                {
                    return null;
                }

                return new TokenSqlCodeExpr
                {
                    Value = sqlToken
                };
            };
        }
    }
}