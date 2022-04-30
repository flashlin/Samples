using System.Collections.Generic;
using System.Linq;
using T1.CodeDom.Core;
using T1.CodeDom.TSql;
using T1.CodeDom.TSql.Expressions;

namespace T1.CodeDom.TSql.Parselets
{
    public class InsertParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            var intoStr = string.Empty;
            if (parser.Scanner.TryConsume(SqlToken.Into, out var intoToken))
            {
                intoStr = parser.Scanner.GetSpanString(intoToken);
            }

            var tableName = parser.ConsumeTableName();
            
            var withExpr = parser.ParsePrefix(SqlToken.With);
            
            var columnList = GetColumnsList(parser);

            if (parser.Scanner.TryConsumeAny(out var execSpan, SqlToken.Exec, SqlToken.Execute))
            {
                var execExpr = parser.PrefixParse(execSpan) as SqlCodeExpr;
                return new InsertIntoFromSqlCodeExpr
                {
                    Table = tableName,
                    ColumnsList = columnList,
                    WithExpr = withExpr,
                    SelectFromExpr = execExpr,
                };
            }

            var outputList = parser.GetOutputListExpr();
            var outputInto = parser.GetOutputIntoExpr();


            var hasGroup = parser.MatchToken(SqlToken.LParen);
            if (parser.Scanner.TryConsume(SqlToken.Select, out var selectToken))
            {
                var selectExpr = new SelectParselet().Parse(selectToken, parser) as SqlCodeExpr;
                if (hasGroup)
                {
                    parser.MatchToken(SqlToken.RParen);
                    selectExpr = new GroupSqlCodeExpr
                    {
                        InnerExpr = selectExpr
                    };
                }

                return new InsertIntoFromSqlCodeExpr
                {
                    Table = tableName,
                    ColumnsList = columnList,
                    OutputList = outputList,
                    OutputIntoExpr = outputInto,
                    SelectFromExpr = selectExpr,
                };
            }


            parser.Scanner.Consume(SqlToken.Values);
            var valuesList = parser.ConsumeByDelimiter(SqlToken.Comma, () =>
            {
                parser.Scanner.Consume(SqlToken.LParen);
                var values = new List<SqlCodeExpr>();
                do
                {
                    var expr = parser.ParseExpIgnoreComment();
                    values.Add(expr);
                } while (parser.Scanner.Match(SqlToken.Comma));

                parser.Scanner.Consume(SqlToken.RParen);

                return new ExprListSqlCodeExpr
                {
                    Items = values.ToList()
                };
            }).ToList();

            return new InsertSqlCodeExpr
            {
                IntoStr = intoStr,
                TableName = tableName,
                Columns = columnList,
                WithExpr = withExpr,
                ValuesList = valuesList
            };
        }

        private static List<string> GetColumnsList(IParser parser)
        {
            var columns = new List<string>();
            if (parser.Scanner.Match(SqlToken.LParen))
            {
                do
                {
                    //var columnStr = parser.ConsumeTokenStringAny(SqlToken.Comma, SqlToken.Identifier, SqlToken.SqlIdentifier);
                    var columnSpan = parser.ConsumeIdentifierToken();
                    var columnStr = parser.Scanner.GetSpanString(columnSpan);
                    columns.Add(columnStr);
                } while (parser.MatchToken(SqlToken.Comma));

                parser.Scanner.Consume(SqlToken.RParen);
            }

            return columns;
        }
    }
}