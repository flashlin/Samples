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

            var keyType = parser.ConsumeAny(SqlParserExtension.ParsePrimaryKey, ParseUnique);

            var clusterExpr = parser.ParseClustered();
            var withExpr = parser.ParseConstraintWithOptions();
            var onPrimary = parser.ParseOnPrimary();

            return new ConstraintSqlCodeExpr
            {
                ConstraintName = constraintName,
                KeyType = keyType,
                ClusterExpr = clusterExpr,
                WithExpr = withExpr,
                OnPrimary = onPrimary,
            };
        }

        private static UniqueKeySqlCodeExpr ParseUnique(IParser parser)
        {
            if (!parser.MatchTokenList(SqlToken.UNIQUE))
            {
                return null;
            }

            var isNonClustered = parser.MatchToken(SqlToken.NONCLUSTERED);
            var uniqueColumnList = new List<OrderItemSqlCodeExpr>();
            if (parser.MatchToken(SqlToken.LParen))
            {
                uniqueColumnList = parser.ParseOrderItemList();
                parser.ConsumeToken(SqlToken.RParen);
            }

            return new UniqueKeySqlCodeExpr
            {
                IsNonClustered = isNonClustered,
                ColumnList = uniqueColumnList
            };
        }
    }

    public class UniqueKeySqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("UNIQUE");

            if (IsNonClustered)
            {
                stream.Write(" NONCLUSTERED");
            }
            
            if (ColumnList != null && ColumnList.Count > 0)
            {
                stream.Write("(");
                ColumnList.WriteToStream(stream);
                stream.Write(")");
            }
        }

        public List<OrderItemSqlCodeExpr> ColumnList { get; set; }
        public bool IsNonClustered { get; set; }
    }

    public class ClusteredSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write($"{ClusterType.ToUpper()}");
            if (ColumnList != null && ColumnList.Count > 0)
            {
                stream.Write("(");
                ColumnList.WriteToStreamWithComma(stream);
                stream.Write(")");
            }
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