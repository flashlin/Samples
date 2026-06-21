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

        return sql.ToString();
    }
}
