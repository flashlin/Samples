using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

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

            var withExpr = parser.ParseConstraintWithOptions();

            return new ConstraintSqlCodeExpr
            {
                ConstraintName = constraintName,
                KeyType = keyType,
                ClusterType = clusterType,
                ColumnList = columnList,
                WithExpr = withExpr
            };
        }
    }

    internal class ConstraintWithSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("WITH(");
            OptionList.WriteToStreamWithComma(stream);
            stream.Write(")");
        }

        public List<SqlCodeExpr> OptionList { get; set; }
    }

    public class FillfactorSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("FILLFACTOR = ");
            Value.WriteToStream(stream);
        }

        public SqlCodeExpr Value { get; set; }
    }
}