using SqlSharpLit.Common.ParserLit.Expressions;
using T1.Standard.IO;

namespace SqlSharpLit.Common.ParserLit;

public class SqlDataType : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.DataType;
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