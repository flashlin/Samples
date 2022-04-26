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

            var clusterExpr = ParseClustered(parser);
            var withExpr = parser.ParseConstraintWithOptions();

            return new ConstraintSqlCodeExpr
            {
                ConstraintName = constraintName,
                KeyType = keyType,
                ClusterExpr = clusterExpr,
                WithExpr = withExpr
            };
        }

        private static ClusteredSqlCodeExpr ParseClustered(IParser parser)
        {
            if (!parser.TryConsumeTokenAny(out var headSpan, SqlToken.CLUSTERED, SqlToken.NONCLUSTERED))
            {
                return null;
            }
            var clusterType = parser.Scanner.GetSpanString(headSpan);
            parser.ConsumeToken(SqlToken.LParen);
            var columnList = parser.ParseOrderItemList();
            parser.ConsumeToken(SqlToken.RParen);
            return new ClusteredSqlCodeExpr
            {
                ClusterType = clusterType,
                ColumnList = columnList,
            };
        }
    }

    public class ClusteredSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write($"{ClusterType.ToUpper()}");
            stream.Write("(");
            ColumnList.WriteToStreamWithComma(stream);
            stream.Write(")");
        }

        public string ClusterType { get; set; }
        public List<OrderItemSqlCodeExpr> ColumnList { get; set; }
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