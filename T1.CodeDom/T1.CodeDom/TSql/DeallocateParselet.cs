using T1.CodeDom.Core;

namespace T1.CodeDom.TSql
{
    public class DeallocateParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            var name = parser.ParseExpIgnoreComment();
            return new DeallocateSqlCodeExpr
            {
                Name = name,
            };
        }
    }
}