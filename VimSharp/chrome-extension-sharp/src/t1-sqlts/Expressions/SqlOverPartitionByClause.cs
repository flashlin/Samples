using T1.Standard.IO;

namespace T1.SqlSharp.Expressions;

public class SqlOverPartitionByClause : ISqlExpression
{
    public SqlType SqlType { get; set; } = SqlType.OverPartitionByClause;
    public TextSpan Span { get; set; } = new();
    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_OverPartitionByClause(this);
    }

    public ISqlExpression Field { get; set; } = new SqlFieldExpr();
    public List<ISqlExpression> By { get; set; } = [];
    public List<SqlOrderColumn> Columns { get; set; } = [];
    public string ToSql()
    {
        var sql = new IndentStringBuilder();
        sql.Write(Field.ToSql());
        sql.Write(" OVER (");
        sql.Write($"PARTITION BY ");
        sql.Write(string.Join(",", By.Select(x => x.ToSql())));
        sql.Write("ORDER BY ");
        foreach (var column in Columns.Select((value,index)=> new {value, index}))
        {
            sql.Write(column.value.ToSql());
            if (column.index < Columns.Count - 1)
            {
                sql.Write(", ");
            }
        }
        sql.Write(")");
        return sql.ToString();
    }
}