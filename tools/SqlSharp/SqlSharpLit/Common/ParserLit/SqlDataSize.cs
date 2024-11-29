using SqlSharpLit.Common.ParserLit.Expressions;
using T1.Standard.IO;

namespace SqlSharpLit.Common.ParserLit;

public class SqlDataSize : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.DataSize;
    public string Size { get; set; } = string.Empty;
    public int Scale { get; set; }
    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        if (!string.IsNullOrEmpty(Size))
        {
            sql.Write("(");
            sql.Write($"{Size}");
            if (Scale > 0)
            {
                sql.Write($", {Scale}");
            }
            sql.Write(")");
        }

        return sql.ToString();
    }
}