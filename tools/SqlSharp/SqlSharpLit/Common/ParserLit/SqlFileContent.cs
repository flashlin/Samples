using T1.SqlSharp.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SqlFileContent
{
    public static SqlFileContent Empty => new();
    public string FileName { get; set; } = string.Empty;
    public string Sql { get; set; } = string.Empty;
    public List<ISqlExpression> SqlExpressions { get; set; } = [];
}