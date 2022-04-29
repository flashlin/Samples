using T1.CodeDom.Core;

namespace T1.CodeDom.TSql.Parselets
{
    public class BatchRefFileParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            var line = parser.Scanner.GetSpanString(token);
            return new BatchReferenceFileSqlCodeExpr
            {
                File = line.Substring(3)
            };
        }
    }
}