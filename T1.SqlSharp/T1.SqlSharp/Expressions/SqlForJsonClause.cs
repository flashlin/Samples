using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlForJsonClause : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.ForJsonClause;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ForJsonClause(this);
    }

    public SqlForJsonMode Mode { get; set; }
    public bool HasRoot { get; set; }
    public ISqlExpression? RootName { get; set; }
    public bool IncludeNullValues { get; set; }
    public bool WithoutArrayWrapper { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write($"FOR JSON {Mode.ToString().ToUpper()}");
        sql.Write(string.Join("", GetDirectives().Select(x => $", {x}")));
        return sql.ToString();
    }

    private IEnumerable<string> GetDirectives()
    {
        if (HasRoot)
        {
            yield return RootName != null ? $"ROOT({RootName.ToSql()})" : "ROOT";
        }
        if (IncludeNullValues)
        {
            yield return "INCLUDE_NULL_VALUES";
        }
        if (WithoutArrayWrapper)
        {
            yield return "WITHOUT_ARRAY_WRAPPER";
        }
    }
}
