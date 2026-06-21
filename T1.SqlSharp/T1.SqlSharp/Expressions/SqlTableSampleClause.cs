using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlTableSampleClause : ISqlExpression
{
    public SqlType SqlType { get; } = SqlType.TableSampleClause;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_TableSampleClause(this);
    }

    public bool IsSystem { get; set; }
    public required ISqlExpression SampleNumber { get; set; }
    public SqlTableSampleUnit Unit { get; set; } = SqlTableSampleUnit.Unspecified;
    public ISqlExpression? RepeatableSeed { get; set; }

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("TABLESAMPLE");
        if (IsSystem)
        {
            sql.Write(" SYSTEM");
        }

        sql.Write($" ({SampleNumber.ToSql()}{GetUnitSql()})");
        if (RepeatableSeed != null)
        {
            sql.Write($" REPEATABLE ({RepeatableSeed.ToSql()})");
        }

        return sql.ToString();
    }

    private string GetUnitSql()
    {
        return Unit switch
        {
            SqlTableSampleUnit.Percent => " PERCENT",
            SqlTableSampleUnit.Rows => " ROWS",
            _ => string.Empty
        };
    }
}
