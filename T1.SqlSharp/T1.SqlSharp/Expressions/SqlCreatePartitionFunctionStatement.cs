using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreatePartitionFunctionStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreatePartitionFunctionStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreatePartitionFunctionStatement(this);
    }

    public string FunctionName { get; set; } = string.Empty;
    public string InputType { get; set; } = string.Empty;
    public string RangeDirection { get; set; } = string.Empty;
    public List<string> BoundaryValues { get; set; } = [];

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"CREATE PARTITION FUNCTION {FunctionName} ({InputType}) AS RANGE");
        if (!string.IsNullOrEmpty(RangeDirection))
        {
            sql.Append($" {RangeDirection}");
        }

        sql.Append($" FOR VALUES ({string.Join(", ", BoundaryValues)})");
        return sql.ToString();
    }
}
