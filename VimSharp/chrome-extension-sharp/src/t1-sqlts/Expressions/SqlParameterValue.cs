using System.Text;
using Microsoft.Extensions.Primitives;

namespace T1.SqlSharp.Expressions;

public class SqlParameterValue : ISqlExpression
{
    public SqlType SqlType => SqlType.ParameterValue;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ParameterValue(this);
    }

    public string Name { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;
    public string ToSql()
    {
        return $@"{Name}={Value}";
    }
}


public class SqlChangeTableChanges : ISqlExpression
{
    public SqlType SqlType => SqlType.ChangeTable;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ChangeTableChanges(this);
    }
    public string TableName { get; set; } = string.Empty;
    public required ISqlExpression LastSyncVersion { get; set; }
    public bool IsForceSeek { get; set; } = false;
    public string Alias { get; set; } = string.Empty;

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CHANGETABLE(CHANGES {TableName}, {LastSyncVersion.ToSql()}");
        if (IsForceSeek)
        {
            sql.Append(", FORCESEEK");
        }
        sql.Append(")");
        if (string.IsNullOrEmpty(Alias))
        {
            sql.Append($" AS {Alias}");
        }
        return sql.ToString();
    }
}

public class SqlChangeTableVersion : ISqlExpression
{
    public SqlType SqlType => SqlType.ChangeTable;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ChangeTableVersion(this);
    }
    public string TableName { get; set; } = string.Empty;
    public List<ISqlExpression> PrimaryKeyValues { get; set; } = [];
    public bool IsForceSeek { get; set; } = false;
    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CHANGETABLE(VERSION {TableName}, ");
        sql.Append(string.Join(",", PrimaryKeyValues.Select(x => x.ToSql())));
        if (IsForceSeek)
        {
            sql.Append(", FORCESEEK");
        }
        sql.Append(")");
        return sql.ToString();
    }
}
