using T1.CodeDom.TSql.Expressions;
using T1.Standard.IO;

namespace T1.CodeDom.TSql.Parselets
{
    public class CreateUserSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("CREATE USER ");
            UserName.WriteToStream(stream);
            stream.Write(" ");
            LoginName.WriteToStream(stream);
            if (WithExpr != null)
            {
                stream.Write(" ");
                WithExpr.WriteToStream(stream);
            }
        }

        public SqlCodeExpr UserName { get; set; }
        public SqlCodeExpr LoginName { get; set; }
        public SqlCodeExpr WithExpr { get; set; }
    }
}