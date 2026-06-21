using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlMergeInsertAction : ISqlMergeAction
{
    public SqlType SqlType => SqlType.MergeInsertAction;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_MergeInsertAction(this);
    }

    public List<string> Columns { get; set; } = [];
    public List<ISqlExpression> Values { get; set; } = [];
    public bool IsDefaultValues { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append("INSERT");
        if (Columns.Count > 0)
        {
            sql.Append($" ({string.Join(", ", Columns)})");
        }
        if (IsDefaultValues)
        {
            sql.Append(" DEFAULT VALUES");
        }
        else
        {
            sql.Append($" VALUES ({string.Join(", ", Values.Select(v => v.ToSql()))})");
        }
        return sql.ToString();
    }
}
