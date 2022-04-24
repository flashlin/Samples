using T1.CodeDom.Core;

namespace T1.CodeDom.TSql.Parselets
{
    public class DeletedParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            parser.ConsumeToken(SqlToken.Dot);
            var columnExpr = parser.Consume(SqlToken.Identifier);

            return new DeletedColumnSqlCodeExpr
            {
                ColumnExpr = columnExpr
            };
        }
    }
}