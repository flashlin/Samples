using System;
using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class DbccParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            return parser.ConsumeAny(ParseDbccUpdateUsage,
                ParseDbccCheckIdent,
                ParseDbccInputbuffer);
        }


        private DbccSqlCodeExpr ParseDbccInputbuffer(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.INPUTBUFFER))
            {
                return null;
            }

            var parametersList = parser.ParseParameterList("INPUTBUFFER", 1, 1);
            
            var withList = new List<SqlCodeExpr>();
            if (parser.MatchToken(SqlToken.With))
            {
                withList = parser.ParseAll(ParseSqlTokenFn(SqlToken.NO_INFOMSGS),
                    ParseSqlTokenFn(SqlToken.COUNT_ROWS));
            }

            return new DbccSqlCodeExpr
            {
                Name = "INPUTBUFFER",
                ParametersList = parametersList,
                WithList = withList
            };
        }

        private DbccUpdateusageSqlCodeExpr ParseDbccUpdateUsage(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.UpdateUsage))
            {
                return null;
            }
            
            var objectIdList = parser.ParseParameterList("UPDATEUSAGE", 1, int.MaxValue);

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

        private DbccSqlCodeExpr ParseDbccCheckIdent(IParser parser)
        {
            if (!parser.MatchToken(SqlToken.CHECKIDENT))
            {
                return null;
            }

            parser.ConsumeToken(SqlToken.LParen);

            var tableName = parser.ConsumeObjectId();
            var optionParametersList = ParseSeqOptionWithComma(
                parser,
                parser1 => parser1.ConsumeToTokenValueAny(SqlToken.NORESEED, SqlToken.RESEED),
                parser1 => parser1.ParseExpIgnoreComment()
            );
            parser.ConsumeToken(SqlToken.RParen);

            var parametersList = new List<SqlCodeExpr>();
            parametersList.Add(tableName);
            parametersList.AddRange(optionParametersList);

            return new DbccSqlCodeExpr
            {
                Name = "CHECKIDENT",
                ParametersList = parametersList
            };
        }

        public static List<SqlCodeExpr> ParseSeqOptionWithComma(IParser parser,
            params Func<IParser, SqlCodeExpr>[] parseList)
        {
            var list = new List<SqlCodeExpr>();
            var index = 0;
            while (parser.MatchToken(SqlToken.Comma))
            {
                var expr = parseList[index](parser);
                list.Add(expr);
                index++;
            }

            return list;
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

    public class DbccSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write($"DBCC {Name}");
            stream.Write("(");
            ParametersList.WriteToStreamWithComma(stream);
            stream.Write(")");

            if (WithList != null && WithList.Count > 0)
            {
                stream.Write(" WITH ");
                WithList.WriteToStream(stream);
            }
        }

        public string Name { get; set; }
        public List<SqlCodeExpr> ParametersList { get; set; }
        public List<SqlCodeExpr> WithList { get; set; }
    }
}