using System;
using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class BreakParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            parser.ConsumeToken(SqlToken.Semicolon);
            return new BreakSqlCodeExpr();
        }
    }

    public class DbccParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            parser.ConsumeToken(SqlToken.Updateusage);
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

    public class DbccUpdateusageSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("DBCC UPDATEUSAGE(");
            ObjectIdList.WriteToStreamWithComma(stream);
            stream.Write(")");
            if (WithList != null && WithList.Count > 0)
            {
                stream.Write(" WITH(");
                WithList.WriteToStreamWithComma(stream);
                stream.Write(")");
            }
        }

        public List<SqlCodeExpr> ObjectIdList { get; set; }
        public List<SqlCodeExpr> WithList { get; set; }
    }
}