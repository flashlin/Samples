using System.Collections.Generic;
using T1.CodeDom.Core;
using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

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

    public class DefaultConstantParselet : IPrefixParselet
    {
        public IExpression Parse(TextSpan token, IParser parser)
        {
            return new DefaultConstantSqlCodeExpr();
        }
    }

    public class DefaultConstantSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("DEFAULT");
        }
    }

    public class DbccUpdateusageSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("DBCC UPDATEUSAGE(");
            ObjectIdList.WriteToStreamWithComma(stream);
            stream.Write(")");
            if (WithList != null && WithList.Count > 0)
            {
                stream.Write(" WITH(");
                WithList.WriteToStreamWithComma(stream);
                stream.Write(")");
            }
        }

        public List<SqlCodeExpr> ObjectIdList { get; set; }
        public List<SqlCodeExpr> WithList { get; set; }
    }
}