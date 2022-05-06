using T1.Standard.IO;

namespace T1.CodeDom.TSql.Expressions
{
    public class AlterDatabaseSqlCodeExpr : SqlCodeExpr
    {
        public override void WriteToStream(IndentStream stream)
        {
            stream.Write("ALTER DATABASE ");
            DatabaseName.WriteToStream(stream);
            
            stream.Write(" ");
            ActionExpr.WriteToStream(stream);
            
            if (IsSemicolon)
            {
                stream.Write(" ;");
            }
        }

        public SqlCodeExpr DatabaseName { get; set; }
        public bool IsSemicolon { get; set; }
        public SqlCodeExpr ActionExpr { get; set; }
    }
}