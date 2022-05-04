using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
	public class VariableParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            var tokenStr = parser.Scanner.GetSpanString(token);
            var variableExpr = new VariableSqlCodeExpr
            {
                Name = tokenStr
            };

            if (parser.MatchToken(SqlToken.Dot))
            {
                var nodesExpr = parser.PrefixParse(SqlToken.NODES) as SqlCodeExpr;
                parser.ConsumeToken(SqlToken.As);
                var aliasName = parser.ConsumeObjectId();
                parser.ConsumeToken(SqlToken.LParen);
                var columnName = parser.ConsumeObjectId();
                parser.ConsumeToken(SqlToken.RParen);
                
                return new VariableNodesSqlCodeExpr
                {
                    Variable = variableExpr,
                    NodesExpr = nodesExpr,
                    AliasName = aliasName,
                    ColumnName = columnName
                };
            }

            return variableExpr;
        }
    }

    public class VariableNodesSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            Variable.WriteToStream(stream);
            stream.Write(".");
            NodesExpr.WriteToStream(stream);
            stream.Write(" AS ");
            AliasName.WriteToStream(stream);
            stream.Write("(");
            ColumnName.WriteToStream(stream);
            stream.Write(")");
        }

        public VariableSqlCodeExpr Variable { get; set; }
        public SqlCodeExpr NodesExpr { get; set; }
        public SqlCodeExpr AliasName { get; set; }
        public SqlCodeExpr ColumnName { get; set; }
    }
}