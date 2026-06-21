using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateViewStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateViewStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateViewStatement(this);
    }

    public bool IsOrAlter { get; set; }
    public bool IsAlter { get; set; }
    public string ViewName { get; set; } = string.Empty;
    public List<string> ColumnNames { get; set; } = [];
    public List<string> Options { get; set; } = [];
    public required ISqlExpression Query { get; set; }
    public bool WithCheckOption { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append(DefinitionLead.ToSql(IsAlter, IsOrAlter, "VIEW"));
        sql.Append(ViewName);
        if (ColumnNames.Count > 0)
        {
            sql.Append($" ({string.Join(", ", ColumnNames)})");
        }
        if (Options.Count > 0)
        {
            sql.Append($" WITH {string.Join(", ", Options)}");
        }
        sql.Append($" AS {Query.ToSql()}");
        if (WithCheckOption)
        {
            sql.Append(" WITH CHECK OPTION");
        }
        return sql.ToString();
    }
}
