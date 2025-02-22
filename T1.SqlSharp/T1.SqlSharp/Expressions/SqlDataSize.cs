using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlDataSize : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.DataSize;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_DataSize(this);
    }

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