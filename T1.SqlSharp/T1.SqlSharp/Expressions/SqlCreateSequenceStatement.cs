using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateSequenceStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateSequenceStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateSequenceStatement(this);
    }

    public string SequenceName { get; set; } = string.Empty;
    public string DataType { get; set; } = string.Empty;
    public ISqlExpression? StartWith { get; set; }
    public ISqlExpression? IncrementBy { get; set; }
    public ISqlExpression? MinValue { get; set; }
    public ISqlExpression? MaxValue { get; set; }
    public ISqlExpression? CacheSize { get; set; }
    public bool IsNoMinValue { get; set; }
    public bool IsNoMaxValue { get; set; }
    public bool IsCycle { get; set; }
    public bool IsNoCycle { get; set; }
    public bool IsCache { get; set; }
    public bool IsNoCache { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE SEQUENCE {SequenceName}");
        if (!string.IsNullOrEmpty(DataType))
        {
            sql.Append($" AS {DataType}");
        }

        if (StartWith != null)
        {
            sql.Append($" START WITH {StartWith.ToSql()}");
        }

        if (IncrementBy != null)
        {
            sql.Append($" INCREMENT BY {IncrementBy.ToSql()}");
        }

        AppendMinMax(sql);
        AppendCycle(sql);
        AppendCache(sql);
        return sql.ToString();
    }

    private void AppendMinMax(StringBuilder sql)
    {
        if (MinValue != null)
        {
            sql.Append($" MINVALUE {MinValue.ToSql()}");
        }
        else if (IsNoMinValue)
        {
            sql.Append(" NO MINVALUE");
        }

        if (MaxValue != null)
        {
            sql.Append($" MAXVALUE {MaxValue.ToSql()}");
        }
        else if (IsNoMaxValue)
        {
            sql.Append(" NO MAXVALUE");
        }
    }

    private void AppendCycle(StringBuilder sql)
    {
        if (IsCycle)
        {
            sql.Append(" CYCLE");
        }
        else if (IsNoCycle)
        {
            sql.Append(" NO CYCLE");
        }
    }

    private void AppendCache(StringBuilder sql)
    {
        if (CacheSize != null)
        {
            sql.Append($" CACHE {CacheSize.ToSql()}");
        }
        else if (IsCache)
        {
            sql.Append(" CACHE");
        }
        else if (IsNoCache)
        {
            sql.Append(" NO CACHE");
        }
    }
}
