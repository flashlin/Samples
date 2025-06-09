using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlDataTypeWithSize : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.DataTypeWithSize;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_DataTypeWithSize(this);
    }

    public string DataTypeName { get; set; } = string.Empty;
    public SqlDataSize Size { get; set; } = new();
    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(DataTypeName);
        sql.Write(Size.ToSql());
        return sql.ToString();
    }
}