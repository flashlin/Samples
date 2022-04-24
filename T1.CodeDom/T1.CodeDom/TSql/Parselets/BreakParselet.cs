using T1.CodeDom.Core;

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
}