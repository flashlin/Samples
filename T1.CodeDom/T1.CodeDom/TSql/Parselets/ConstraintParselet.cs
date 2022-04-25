using T1.CodeDom.Core;

namespace T1.CodeDom.TSql.Parselets
{
    public class ConstraintParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            var constraintName = parser.ConsumeObjectId();
            var keyType = parser.ConsumeTokenStringListAny(new[] {SqlToken.PRIMARY, SqlToken.KEY},
                new[] {SqlToken.UNIQUE});
            var clusterType = parser.ConsumeTokenAny(SqlToken.CLUSTERED, SqlToken.NONCLUSTERED).Type;

            parser.ConsumeToken(SqlToken.LParen);
            var columnList = parser.ParseOrderItemList();
            parser.ConsumeToken(SqlToken.RParen);

            return new ConstraintSqlCodeExpr
            {
                ConstraintName = constraintName,
                KeyType = keyType,
                ClusterType = clusterType,
                ColumnList = columnList
            };
        }
    }
}