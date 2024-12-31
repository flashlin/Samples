using T1.SqlSharp.Expressions;

namespace SqlSharpLit.Common.ParserLit;

public class SqlSelectContent
{
    public static SqlSelectContent Empty => new();
    public string FileName { get; set; } = string.Empty;
    public List<SelectStatement> Statements { get; set; } = [];

    public bool HasSelectSql()
    {
        return this != Empty && Statements.Count > 0;
    }
}