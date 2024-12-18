using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlPartitionBy : ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.PartitionBy;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_PartitionBy(this);
    }

    public List<ISqlExpression> Columns { get; set; } = [];

    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write("PARTITION BY ");
        foreach (var column in Columns.Select((value,index)=> new {value, index}))
        {
            sql.Write(column.value.ToSql());
            if (column.index < Columns.Count - 1)
            {
                sql.Write(", ");
            }
        }
        return sql.ToString();
    }
}