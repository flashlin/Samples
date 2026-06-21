using System.Text;

namespace T1.SqlSharp.Expressions;

public class SqlCreateFunctionStatement : ISqlExpression
{
    public SqlType SqlType => SqlType.CreateFunctionStatement;
    public TextSpan Span { get; set; } = new();

    public void Accept(SqlVisitor visitor)
    {
        visitor.Visit_CreateFunctionStatement(this);
    }

    public bool IsOrAlter { get; set; }
    public bool IsAlter { get; set; }
    public string FunctionName { get; set; } = string.Empty;
    public List<SqlProcedureParameter> Parameters { get; set; } = [];
    public string ReturnType { get; set; } = string.Empty;
    public SqlDataSize? ReturnSize { get; set; }
    public string ReturnTableVariable { get; set; } = string.Empty;
    public List<SqlColumnDefinition> ReturnTableColumns { get; set; } = [];
    public List<string> Options { get; set; } = [];
    public required ISqlExpression Body { get; set; }

    public string ToSql()
    {
        var sql = new StringBuilder();
        sql.Append(DefinitionLead.ToSql(IsAlter, IsOrAlter, "FUNCTION"));
        sql.Append(FunctionName);
        sql.Append($" ({string.Join(", ", Parameters.Select(p => $"{p.Name} {p.DataType}"))})");
        sql.Append($" RETURNS {ReturnType}");
        if (ReturnSize != null)
        {
            sql.Append(ReturnSize.ToSql());
        }
        if (Options.Count > 0)
        {
            sql.Append($" WITH {string.Join(", ", Options)}");
        }
        sql.Append($" AS {Body.ToSql()}");
        return sql.ToString();
    }
}
