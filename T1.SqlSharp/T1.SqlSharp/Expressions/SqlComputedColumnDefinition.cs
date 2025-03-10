using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlComputedColumnDefinition : ISqlExpression
{
    public string ColumnName { get; set; } = string.Empty;
    public string Expression { get; set; } = string.Empty;
    public bool IsPersisted { get; set; } = false;
    public bool IsNotNull { get; set; } = false;
    public SqlType SqlType { get; } = SqlType.ComputedColumn;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_ComputedColumnDefinition(this);
    }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append($"{ColumnName} AS {Expression}");
        if (IsPersisted)
        {
            sql.Append(" PERSISTED");
        }

        if (IsNotNull)
        {
            sql.Append(" NOT NULL");
        }

        return sql.ToString();
    }
}