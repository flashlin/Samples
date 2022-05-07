using T1.CodeDom.Core;

namespace T1.CodeDom.TSql.Parselets
{
    public class CloseParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            var name = parser.ParseExpIgnoreComment(int.MaxValue);
            return new CloseSqlCodeExpr
            {
                Name = name
            };
        }
    }
}