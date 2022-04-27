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
            stream.Write("ADD FILEGROUP ");
            FileGroupName.WriteToStream(stream);
            if (IsSemicolon)
            {
                stream.Write(" ;");
            }
        }

        public SqlCodeExpr DatabaseName { get; set; }
        public SqlCodeExpr FileGroupName { get; set; }
        public bool IsSemicolon { get; set; }
    }
}