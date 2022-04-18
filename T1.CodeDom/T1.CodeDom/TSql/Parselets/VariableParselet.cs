using PreviewLibrary.Pratt.TSql.Expressions;
using T1.CodeDom.Core;

namespace T1.CodeDom.TSql.Parselets
{
	public class VariableParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            var tokenStr = parser.Scanner.GetSpanString(token);
            return new VariableSqlCodeExpr
            {
                Name = tokenStr
            };
        }
    }
}